# 🗑️ WasteWise: Smart Bin Prediction & Route Optimization

> A Machine Learning project that predicts which waste bins are about to overflow
> and plans the most efficient garbage collection route — no IoT hardware needed!

---

## 📌 Features

- **Synthetic Data Generation** — Simulates a city grid of 30–150 smart waste bins with realistic fill levels, locations, and overflow history
- **ML-Powered Overflow Prediction** — Decision Tree Classifier predicts which bins are likely to overflow based on current fill level and past behavior
- **Active Bin Selection** — Flags high-risk bins above a configurable probability threshold (default: 60%)
- **Greedy Route Optimization** — Plans the truck's collection route using the Nearest Neighbor heuristic (always go to the closest unvisited bin)
- **OpenCV Route Visualization** — Draws a city map with all bins, highlights active bins in red, and overlays the optimized route
- **Interactive Streamlit Dashboard** — Full step-by-step UI with data tables, metrics, and the route image

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Frontend | Streamlit |
| ML Model | Scikit-Learn (Decision Tree) |
| Data | NumPy + Pandas (synthetic) |
| Visualization | OpenCV (`opencv-python-headless`) |
| Language | Python 3.9+ |

---

## 📁 Project Structure

```
WasteWise/
│── app.py               # Streamlit frontend — run this!
│── data_generator.py    # Creates synthetic bin dataset
│── model.py             # Trains and uses the Decision Tree model
│── route_optimizer.py   # Greedy Nearest Neighbor route planner
│── visualization.py     # OpenCV map drawing
│── requirements.txt     # All Python dependencies
│── README.md            # This file
```

---

## 🚀 How to Run

### 1. Clone or Download the Project
```bash
git clone https://github.com/YOUR_USERNAME/WasteWise.git
cd WasteWise
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🔄 Step-by-Step Usage (in the app)

| Step | Button | What Happens |
|---|---|---|
| 1 | **Generate Data** | Creates a synthetic dataset of waste bins |
| 2 | **Train Model** | Trains a Decision Tree and shows accuracy |
| 3 | **Predict Overflow** | Gives overflow probability for every bin |
| 4 | **Optimize Route** | Plans the truck route and draws the map |

Use the **sidebar sliders** to adjust:
- Number of bins (30–150)
- Overflow probability threshold (0.40–0.90)

---

## 🧠 Project Pipeline Explained

```
[Synthetic Data] → [Decision Tree ML] → [Probability Scores]
                                              ↓
                              [Filter High-Risk Bins > 0.6]
                                              ↓
                         [Greedy Nearest Neighbor Routing]
                                              ↓
                        [OpenCV Map] → [Streamlit Display]
```

### Why Greedy Nearest Neighbor?

The Travelling Salesman Problem (finding the absolute shortest route) is computationally hard (NP-hard). Greedy Nearest Neighbor gives a fast, good-enough solution:
- Start at depot (0, 0)
- Always move to the closest unvisited active bin
- Repeat until all active bins are visited

This is how many real-world route planners work in practice!

---

## 📊 Sample Results

- **Accuracy**: typically 85–95% on synthetic data
- **Active bins**: ~20–35% of total bins flagged (varies by threshold)
- **Route**: automatically ordered to minimize travel distance

---

## 🎓 Learning Outcomes

By studying this project, you'll understand:
- Synthetic dataset creation with NumPy
- Binary classification with Decision Tree
- `train_test_split`, `predict_proba`, `accuracy_score`
- Greedy algorithms for route optimization
- OpenCV for non-camera image drawing
- Streamlit session state and interactive UI

---

## 👤 Author

Built as a BTech final-year project demonstrating applied Machine Learning
for smart city infrastructure.

---

## 📄 License

MIT License — free to use, modify, and share.
