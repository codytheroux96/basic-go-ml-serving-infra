package main

import (
	"encoding/json"
	"log"
	"math"
	"net/http"
)

type PredictRequest struct {
	Input []float32 `json:"input"`
}

type PredictResponse struct {
	Prediction int `json:"prediction"`
}

var predictor *Predictor

func predictHandler(w http.ResponseWriter, r *http.Request) {
	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid Input", http.StatusBadRequest)
		return
	}

	if len(req.Input) != 28*28 {
		http.Error(w, "Input must be a 28x28 flattened array", http.StatusBadRequest)
		return
	}

	outputs, err := predictor.Predict(req.Input)
	if err != nil {
		http.Error(w, "Prediction Failed", http.StatusInternalServerError)
		log.Println("Prediction Error", err)
		return
	}

	maxVal := float32(math.Inf(-1))
	maxIdx := 0
	for i, val := range outputs {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	resp := PredictResponse{Prediction: maxIdx}
	json.NewEncoder(w).Encode(resp)
}

func main() {
	var err error
	predictor, err = NewPredictor("models/mnist-12.onnx")
	if err != nil {
		log.Fatal("Failed to laod model:", err)
	}

	http.HandleFunc("/predict", predictHandler)
	log.Println("Server running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
