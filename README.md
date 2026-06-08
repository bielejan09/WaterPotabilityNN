# Water Potability Prediction: Neural Network Classifier

Predicts drinking water safety from 9 physical and chemical measurements 
using a PyTorch MLP Neural Network and Random Forest classifier deployed 
as a live REST API on Microsoft Azure.

**Course:** Artificial Intelligence Development and Security  
**University:** National Taipei University of Technology (NTUT)  
**Student:** Jan Pawel Bielecki | ID: 114012111,  Victoria Kammerer | ID: 114012110,  Winslet Clarisse | ID: 114012120

**SDG:** SDG 6 — Clean Water and Sanitation  

---

## Live Demo

Web interface: http://72.155.73.40:5000  
API endpoint:  http://72.155.73.40:5000/predict  
Health check:  http://72.155.73.40:5000/health  

---

## Repository Structure
```bash
WaterPotabilityNN/
├── WaterPotabilityNN.ipynb          # Main notebook: full ML pipeline
├── app.py                           # Flask API serving both models
├── requirements.txt                 # Python dependencies
├── water_potability.csv             # Dataset (Kadiwal, 2021)
├── water_potability_model.pkl       # Trained Random Forest (threshold 0.6)
├── pytorch_model.pth                # Trained PyTorch MLP v1 (threshold 0.7)
├── pytorch_model_info.json          # PyTorch architecture and threshold info
├── scaler.pkl                       # Fitted StandardScaler
├── threshold.pkl                    # RF optimal threshold (0.6)
├── model_card_water_potability.json # MLSecOps model card
└── data_card_water_potability.json  # MLSecOps data card
```
---

## Building Instructions

### Option 1: Use the Live Web Interface (easiest)

No installation required. Open your browser and go to:

http://72.155.73.40:5000

Enter 9 water measurements and click **Predict**. The API returns predictions 
from both the Neural Network (threshold 0.7) and Random Forest (threshold 0.6) 
with a dynamic recommendation based on model certainty.

---

### Option 2: Run the Notebook in Google Colab

1. Open the notebook directly:  
   https://colab.research.google.com/github/bielejan09/WaterPotabilityNN/blob/main/WaterPotabilityNN.ipynb

2. Click **Runtime → Run all**

3. The notebook trains all models, evaluates results, and tests the live API.

---

### Option 3: Run Locally

**Requirements:** Python 3.11+, pip

**Step 1: Clone the repository**
```bash
git clone https://github.com/bielejan09/WaterPotabilityNN.git
cd WaterPotabilityNN
```

**Step 2: Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run the Flask API**
```bash
python3 app.py
```

**Step 5: Open in browser**

http://localhost:5000

---

### Option 3: Deploy to Azure VM

**Requirements:** Azure VM running Ubuntu 22.04+, port 5000 open

**Step 1: Connect to VM via SSH**
```bash
ssh username@your-vm-ip
```

**Step 2: Install dependencies**
```bash
sudo apt update && sudo apt install -y python3 python3-venv git
```

**Step 3: Clone and set up**
```bash
git clone https://github.com/bielejan09/WaterPotabilityNN.git
cd WaterPotabilityNN
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 4: Run permanently with nohup**
```bash
nohup python3 app.py &
```

**Step 5: Update after changes**
```bash
git pull
pkill -f "python3 app.py"
nohup python3 app.py &
```

---

## API Usage

Send a POST request with 9 water measurements in this order:

`[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]`

**Request:**
```json
POST /predict
Content-Type: application/json

{
  "features": [7.2, 204.0, 20791.0, 7.3, 368.0, 564.0, 10.3, 86.0, 2.96]
}
```

**Response:**
```json
{
  "neural_network": {
    "label": "Non-potable",
    "confidence": 0.521,
    "threshold": 0.7,
    "safe": false
  },
  "random_forest": {
    "label": "Non-potable",
    "confidence": 0.355,
    "threshold": 0.6,
    "safe": false
  },
  "recommended": {
    "model": "Neural Network (PyTorch v1)",
    "reason": "Higher certainty on this specific sample (distance from 0.5)",
    "label": "Non-potable",
    "safe": false
  }
}
```

---

## Model Performance

| Model | Accuracy | Recall (Unsafe) | False Negatives |
|---|:---:|:---:|:---:|
| TF v1 / PyTorch v1 (0.5) | 64.3% | 0.750 | 100 |
| PyTorch v2 | 61.9% | 0.630 | 148 |
| PyTorch v3 | 60.5% | 0.642 | 143 |
| **PyTorch v1 (0.7)** | **67.7%** | **0.952** | **19** |
| Random Forest (0.5) | 64.0% | 0.770 | 92 |
| Random Forest (0.6) | 66.8% | 0.932 | 27 |

PyTorch v1 with threshold 0.7 is the best performing model, 
achieving 95.2% recall on unsafe water with only 19 missed samples.

---

## Paper Study

1. Saroja & Haseena (2023). Deep learning approach for prediction and classification of potable water. *Analytical Sciences*, 39, 1179–1189. https://doi.org/10.1007/s44211-023-00328-2

2. Ainapure et al. (2023). Drinking water potability prediction using machine learning approaches. *Water Practice & Technology*, 18(12), 3004. https://doi.org/10.2166/wpt.2023.202

3. Grinsztajn et al. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS 2022*. https://arxiv.org/abs/2207.08815

---

## Security Notes

This project was deployed on a public Azure VM. Within hours of deployment, 
165 automated credential scanning requests were received from a known scanning 
service (Advin Services LLC, AS22295). All requests returned 404 —> no 
credentials were exposed. This incident is documented in the security 
discussion as real-world evidence of infrastructure threats against deployed 
AI systems.

---

## Dataset

Kadiwal, A. (2021). Water Potability Dataset. Kaggle.  
https://www.kaggle.com/datasets/adityakadiwal/water-potability

3,276 samples, 9 features, binary label (0 = Non-potable, 1 = Potable).
