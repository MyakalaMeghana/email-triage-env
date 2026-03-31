from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"App is running")

port = 7860
server = HTTPServer(("0.0.0.0", port), Handler)

print(f"Server running on port {port}")
server.serve_forever()
