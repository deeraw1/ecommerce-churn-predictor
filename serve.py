from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sqlite3
import json
from urllib.parse import parse_qs, urlparse
from datetime import datetime

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add headers to prevent caching and enable CORS
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests for CORS
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        try:
            # API routes
            if self.path.startswith('/api/'):
                self.handle_api_request()
            # Serve sample CSV file
            elif self.path == '/sample.csv':
                self.serve_file('sample.csv', 'text/csv')
            # Serve frontend files
            else:
                self.serve_frontend()
                
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
            print(f"Error: {e}")

    def do_POST(self):
        try:
            if self.path.startswith('/api/'):
                self.handle_api_request()
            else:
                self.send_error(404, "Endpoint not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
            print(f"Error: {e}")

    def serve_frontend(self):
        """Serve frontend files"""
        file_path = self.path
        
        # Default to index.html for root or unknown paths
        if file_path == '/' or not os.path.exists('frontend' + file_path):
            file_path = '/index.html'
        
        # Serve from frontend directory
        full_path = os.path.join('frontend', file_path.lstrip('/'))
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            self.serve_file(full_path)
        else:
            self.send_error(404, "File not found")

    def serve_file(self, file_path, content_type=None):
        """Serve a static file"""
        if not os.path.exists(file_path):
            self.send_error(404, "File not found")
            return
            
        if content_type is None:
            # Auto-detect content type
            if file_path.endswith('.html'):
                content_type = 'text/html'
            elif file_path.endswith('.css'):
                content_type = 'text/css'
            elif file_path.endswith('.js'):
                content_type = 'application/javascript'
            elif file_path.endswith('.png'):
                content_type = 'image/png'
            elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            else:
                content_type = 'text/plain'
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        except Exception as e:
            self.send_error(500, f"Error reading file: {str(e)}")

    def handle_api_request(self):
        """Handle API endpoints"""
        if self.path == '/api/health':
            self.handle_health()
        elif self.path == '/api/predictions/history':
            self.handle_history()
        elif self.path == '/api/predict':
            self.handle_predict()
        elif self.path.startswith('/api/predict/upload'):
            self.handle_file_upload()
        else:
            self.send_error(404, "API endpoint not found")

    def handle_health(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "healthy",
            "message": "Server is running smoothly",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "health": "/api/health",
                "predict": "/api/predict",
                "history": "/api/predictions/history",
                "batch_upload": "/api/predict/upload/{csv,excel}"
            }
        }
        self.wfile.write(json.dumps(response).encode())

    def handle_history(self):
        """Get prediction history"""
        try:
            # Parse query parameters
            query = urlparse(self.path).query
            params = parse_qs(query)
            limit = int(params.get('limit', [20])[0])
            
            conn = sqlite3.connect('predictions.db')
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', (limit,))
            rows = c.fetchall()
            conn.close()
            
            history = [dict(row) for row in rows]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "count": len(history),
                "predictions": history,
                "limit": limit
            }
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Database error: {str(e)}")

    def handle_predict(self):
        """Handle single prediction"""
        # For now, just return a message that this should go to FastAPI
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "message": "Single predictions should be handled by FastAPI on port 8001",
            "status": "info"
        }
        self.wfile.write(json.dumps(response).encode())

    def handle_file_upload(self):
        """Handle file uploads"""
        # For now, just return a message that this should go to FastAPI
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "message": "File uploads should be handled by FastAPI on port 8001",
            "status": "info"
        }
        self.wfile.write(json.dumps(response).encode())

def check_database():
    """Ensure database exists and is properly set up"""
    if not os.path.exists('predictions.db'):
        print("⚠️  Database not found. Creating new database...")
        init_database()
    else:
        print("✅ Database found and ready")

def init_database():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenure INTEGER,
            numberofaddress INTEGER,
            cashbackamount REAL,
            daysincelastorder INTEGER,
            ordercount INTEGER,
            satisfactionscore INTEGER,
            churn_prediction INTEGER,
            churn_probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_data TEXT,
            insights TEXT
        )
    ''')
    
    # Create indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_churn ON predictions(churn_prediction)')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

if __name__ == '__main__':
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check and initialize database
    check_database()
    
    print("🚀 Starting E-commerce Churn Predictor Server")
    print("📍 Frontend: http://localhost:8000")
    print("📍 API Health: http://localhost:8000/api/health")
    print("📍 Sample CSV: http://localhost:8000/sample.csv")
    print("")
    print("⚠️  Note: For actual predictions, ensure FastAPI is running on port 8001")
    print("    Command: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001")
    print("")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start server
    server = HTTPServer(('localhost', 8000), CustomHTTPRequestHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        server.server_close()
        print("✅ Server shut down cleanly")