import http.server
import socketserver
import json
import urllib.request
import urllib.error
import os

PORT = 8080
VSCODE_BRIDGE_URL = 'http://127.0.0.1:20129'
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    'openai:gpt-4.1': ('openai', 'gpt-4.1'),
    'openai:gpt-4.1-mini': ('openai', 'gpt-4.1-mini'),
    'anthropic:claude-sonnet-4-20250514': ('anthropic', 'claude-sonnet-4-20250514'),
    'anthropic:claude-opus-4-20250514': ('anthropic', 'claude-opus-4-20250514'),
}
AUTO_MODELS = ('openai:gpt-4.1', 'anthropic:claude-sonnet-4-20250514')
CHAT_SYSTEM_PROMPT = (
    'You are a careful ETH chart-analysis assistant. Use the supplied live dashboard context and OHLCV candles '
    'as the source of truth. Candle timestamps are Unix milliseconds and the supplied candles are 1-minute '
    'ETHUSDT candles. Explain uncertainty, distinguish observed data from inference, and never claim certainty '
    'or promise returns. Do not invent indicators, chart drawings, news, or candles that are not supplied. '
    'The embedded TradingView chart may have a different timeframe or symbol; you can only directly analyze '
    'the supplied ETHUSDT one-minute feed. This is informational analysis, not financial advice.'
)


def available_providers():
    return {
        'openai': bool(os.environ.get('OPENAI_API_KEY')),
        'anthropic': bool(os.environ.get('ANTHROPIC_API_KEY')),
    }


def get_vscode_models():
    try:
        req = urllib.request.Request(f'{VSCODE_BRIDGE_URL}/models')
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.loads(response.read()).get('models', [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []


def resolve_model(model_choice):
    providers = available_providers()
    if model_choice == 'auto':
        for choice in AUTO_MODELS:
            provider, model = MODELS[choice]
            if providers[provider]:
                return provider, model
        raise ValueError('No model API key is configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY on the server, then restart it.')
    if model_choice not in MODELS:
        raise ValueError('Choose Auto or one of the listed OpenAI and Anthropic models.')
    provider, model = MODELS[model_choice]
    if not providers[provider]:
        key_name = 'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'
        raise ValueError(f'{provider.title()} is not configured. Set {key_name} on the server, then restart it.')
    return provider, model


def call_model(provider, model, messages, context):
    context_text = json.dumps(context, separators=(',', ':'), ensure_ascii=True)
    if provider == 'openai':
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': f'{CHAT_SYSTEM_PROMPT}\n\nCurrent dashboard data (JSON):\n{context_text}'}
            ] + messages,
            'max_completion_tokens': 1200,
            'temperature': 0.2,
        }
        headers = {
            'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}",
            'Content-Type': 'application/json',
        }
        url = 'https://api.openai.com/v1/chat/completions'
    else:
        payload = {
            'model': model,
            'max_tokens': 1200,
            'system': f'{CHAT_SYSTEM_PROMPT}\n\nCurrent dashboard data (JSON):\n{context_text}',
            'messages': messages,
        }
        headers = {
            'x-api-key': os.environ['ANTHROPIC_API_KEY'],
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json',
        }
        url = 'https://api.anthropic.com/v1/messages'

    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        try:
            detail = json.loads(detail).get('error', {}).get('message', detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(f'{provider.title()} request failed ({error.code}): {detail[:500]}') from error
    if provider == 'openai':
        return result['choices'][0]['message']['content'], result.get('model', model)
    text = ''.join(block.get('text', '') for block in result.get('content', []) if block.get('type') == 'text')
    return text, result.get('model', model)

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/api/chat/models':
            vscode_models = get_vscode_models()
            self.send_json(200, {
                'providers': {**available_providers(), 'vscodeModels': bool(vscode_models)},
                'models': list(MODELS),
                'vscodeModels': vscode_models,
            })
        elif self.path.startswith('/api/vision/'):
            self.proxy_vision()
        elif self.path.startswith('/api/binance/'):
            self.proxy_binance()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/chat':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 150000:
                raise ValueError('Chat request must be between 1 byte and 150 KB.')
            body = json.loads(self.rfile.read(length))
            model_choice = body.get('model', 'auto')
            vscode_model_id = None
            if model_choice == 'vscode:auto' or (isinstance(model_choice, str) and model_choice.startswith('vscode:')):
                vscode_models = get_vscode_models()
                if not vscode_models:
                    raise ValueError('VS Code models are unavailable. Start the STO VS Code model bridge with F5, then reload this page.')
                if model_choice == 'vscode:auto':
                    vscode_model_id = vscode_models[0]['id']
                else:
                    vscode_model_id = model_choice.removeprefix('vscode:')
                    if not any(model.get('id') == vscode_model_id for model in vscode_models):
                        raise ValueError('That VS Code model is no longer available. Reload the page and choose an available model.')
                provider, model = 'vscode', vscode_model_id
            else:
                provider, model = resolve_model(model_choice)
            messages = body.get('messages', [])
            if not isinstance(messages, list) or len(messages) > 12:
                raise ValueError('Conversation history is invalid or too long.')
            clean_messages = []
            for message in messages:
                role = message.get('role')
                content = message.get('content')
                if role not in ('user', 'assistant') or not isinstance(content, str):
                    raise ValueError('Conversation contains an invalid message.')
                clean_messages.append({'role': role, 'content': content[:8000]})
            question = body.get('question')
            if not isinstance(question, str) or not question.strip():
                raise ValueError('Enter a question about the chart.')
            clean_messages.append({'role': 'user', 'content': question.strip()[:8000]})
            context = body.get('context', {})
            if not isinstance(context, dict):
                raise ValueError('Chart context must be a JSON object.')
            if provider == 'vscode':
                answer, used_model = self.call_vscode_model(model, clean_messages, context)
            else:
                answer, used_model = call_model(provider, model, clean_messages, context)
            self.send_json(200, {'answer': answer, 'model': used_model, 'provider': provider})
        except ValueError as error:
            self.send_json(400, {'error': str(error)})
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            self.send_json(502, {'error': f'Invalid response from model provider: {error}'})
        except (RuntimeError, urllib.error.URLError, TimeoutError) as error:
            self.send_json(502, {'error': str(error)})

    def call_vscode_model(self, model_id, messages, context):
        payload = json.dumps({'modelId': model_id, 'messages': messages, 'context': context}).encode()
        request = urllib.request.Request(
            f'{VSCODE_BRIDGE_URL}/chat',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace')
            try:
                detail = json.loads(detail).get('error', detail)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f'VS Code model request failed: {detail[:500]}') from error
        return result['answer'], result.get('model', model_id)

    def send_json(self, status, data):
        encoded = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def proxy_vision(self):
        vision_path = self.path.replace('/api/vision/', '', 1)
        url = f'https://data-api.binance.vision/api/v3/{vision_path}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def proxy_binance(self):
        binance_path = self.path.replace('/api/binance/', '', 1)
        url = f'https://api.binance.us/api/v3/{binance_path}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def log_message(self, format, *args):
        if '502' in str(args) or 'stock_details' in str(args):
            super().log_message(format, *args)

class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

if __name__ == '__main__':
    with ThreadedServer(('', PORT), ProxyHandler) as httpd:
        print(f'Serving on http://localhost:{PORT}')
        httpd.serve_forever()
