#!/bin/bash
# SmartRoad Quickstart Script
# ==========================
# Run this script to get SmartRoad up and running locally in 5 minutes
# 
# Usage:
#   bash quickstart.sh      # Interactive setup
#   bash quickstart.sh run  # Just start the server

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          🛣️  SmartRoad - AI Street Defect Detection          ║
║                                                                ║
║              Built for Morocco 🇲🇦                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Function to print section headers
print_heading() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Check Python version
print_heading "Checking Python Installation"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo "  Install from: https://www.python.org/downloads/"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION found${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_heading "Creating Virtual Environment"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
print_heading "Activating Virtual Environment"
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
print_heading "Installing Dependencies"
echo "  This may take a few minutes (2-5 GB for PyTorch)..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_heading "Creating Environment Configuration"
    cp .env.example .env
    echo -e "${YELLOW}⚠ Created .env file - you must add your API keys:${NC}"
    echo ""
    echo "  1. Edit .env file:"
    echo "     nano .env"
    echo ""
    echo "  2. Add your API key (choose one):"
    echo "     LLM_PROVIDER=groq"
    echo "     GROQ_API_KEY=your_key_here"
    echo ""
    echo "  3. Get free API keys:"
    echo "     Groq: https://console.groq.com/keys"
    echo "     OpenAI: https://platform.openai.com/account/api-keys"
    echo ""
else
    echo -e "${GREEN}✓ .env file exists${NC}"
    if grep -q "GROQ_API_KEY=" .env; then
        API_KEY=$(grep "GROQ_API_KEY=" .env | cut -d'=' -f2)
        if [ -z "$API_KEY" ] || [ "$API_KEY" == "" ]; then
            echo -e "${YELLOW}⚠ GROQ_API_KEY is empty - you must configure it${NC}"
        fi
    fi
fi

# Check if running with 'run' argument
if [ "$1" == "run" ]; then
    # Skip to running
    true
else
    # Interactive setup complete, ask user
    print_heading "Setup Complete!"
    echo ""
    echo -e "  ${GREEN}✓${NC} Python environment ready"
    echo -e "  ${GREEN}✓${NC} Dependencies installed"
    echo -e "  ${GREEN}✓${NC} Configuration file created"
    echo ""
    echo -e "  ${YELLOW}Next Steps:${NC}"
    echo ""
    echo "  1. Configure API keys:"
    echo "     ${BLUE}nano .env${NC}"
    echo ""
    echo "  2. Start the application:"
    echo "     ${BLUE}bash quickstart.sh run${NC}"
    echo "     or"
    echo "     ${BLUE}uvicorn backend.main:app --reload --port 8000${NC}"
    echo ""
    echo "  3. Open in browser:"
    echo "     ${BLUE}http://localhost:8000${NC}"
    echo ""
    read -p "Start the server now? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Run the application
print_heading "🚀 Starting SmartRoad Backend"
echo ""
echo -e "${BLUE}Starting FastAPI server...${NC}"
echo -e "${YELLOW}Loading YOLO model (this takes 30-60 seconds on first run)...${NC}"
echo ""
echo -e "${GREEN}Application will be available at:${NC}"
echo -e "${BLUE}  http://localhost:8000${NC}"
echo ""
echo -e "${YELLOW}Press CTRL+C to stop${NC}"
echo ""

# Start with uvicorn
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
