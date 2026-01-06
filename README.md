# Intelligent-Heart-Disease-Prediction-System
Intelligent Heart Disease Prediction System
This Project is Built for Bala

Install streamlt and setup proper tunnel.


!pip install -q streamlit
!npm install -g localtunnel > /dev/null 2>&1

import time
print("--------------------------------------------------")
print("👇 COPY THIS IP ADDRESS:")
!wget -q -O - ipv4.icanhazip.com
print("--------------------------------------------------")
print("⏳ Starting VPNP Server (Please wait 5 seconds)...")

# Flags to prevent loading issues
!streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false & npx localtunnel --port 8501
