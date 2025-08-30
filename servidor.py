#!/usr/bin/env python3
"""
Servidor HTTP simples para testar o dashboard local
Execute: python3 servidor.py
Acesse: http://localhost:8000
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000

# Mudança para o diretório do script
os.chdir(Path(__file__).parent)

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Adicionar headers CORS para evitar problemas
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    print(f"🚀 Iniciando servidor HTTP na porta {PORT}")
    print(f"📂 Diretório: {os.getcwd()}")
    print(f"🌐 Acesse: http://localhost:{PORT}")
    print(f"📊 Dashboard: http://localhost:{PORT}/index.html")
    print("🔄 Pressione Ctrl+C para parar")
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"\n✅ Servidor rodando em http://localhost:{PORT}")
            
            # Tentar abrir o navegador automaticamente
            try:
                webbrowser.open(f'http://localhost:{PORT}/index.html')
                print("🌐 Abrindo navegador...")
            except:
                pass
                
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")

if __name__ == "__main__":
    start_server()
