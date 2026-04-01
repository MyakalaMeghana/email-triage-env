from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from env import EmailEnv
from models import Action

env = EmailEnv()

class Handler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/reset":
            state = env.reset()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response = {
                "emails": state.emails,
                "remaining": state.remaining
            }
            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/step":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            action = Action(action_type=data.get("action"))

            state, reward, done, _ = env.step(action)

            response = {
                "emails": state.emails,
                "remaining": state.remaining,
                "reward": reward.score,
                "done": done
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()


server = HTTPServer(("0.0.0.0", 7860), Handler)
print("Server running on port 7860...")
server.serve_forever()
