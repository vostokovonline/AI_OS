#!/bin/bash
# Get WSL2 IP address for Windows access

echo "========================================="
echo "WSL2 Dashboard Access URLs"
echo "========================================="
echo ""

# Get WSL2 IP address
WSL_IP=$(hostname -I | awk '{print $1}')
echo "WSL2 IP Address: $WSL_IP"
echo ""

echo "Access from Windows:"
echo "  Dashboard: http://$WSL_IP:3000"
echo "  Backend API: http://$WSL_IP:8000"
echo ""

echo "Local access within WSL2:"
echo "  Dashboard: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo ""

echo "========================================="
echo "Windows PowerShell Port Forwarding (if needed):"
echo "========================================="
echo ""
echo "netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=$WSL_IP"
echo "netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$WSL_IP"
echo ""
echo "To remove port forwarding later:"
echo "netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0"
echo "netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0"
echo ""
echo "To view all port forwarding rules:"
echo "netsh interface portproxy show all"
echo "========================================="
