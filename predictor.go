package main

import (
	"math"
	"os"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

type Predictor struct {
	model   *onnx.Model
	backend *gorgonnx.Graph
}

func NewPredictor(modelPath string) (*Predictor, error) {
	b, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, err
	}

	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)

	if err := model.UnmarshalBinary(b); err != nil {
		return nil, err
	}

	return &Predictor{model: model, backend: backend}, nil
}

func softmax(logits []float32) []float32 {
	expSum := float32(0.0)
	for _, v := range logits {
		expSum += float32(math.Exp(float64(v)))
	}

	softmaxValues := make([]float32, len(logits))
	for i, v := range logits {
		softmaxValues[i] = float32(math.Exp(float64(v))) / expSum
	}
	return softmaxValues
}

func (p *Predictor) Predict(input []float32) ([]float32, error) {
	t := tensor.New(tensor.WithShape(1, 1, 28, 28), tensor.WithBacking(input))

	if err := p.model.SetInput(0, t); err != nil {
		return nil, err
	}

	if err := p.backend.Run(); err != nil {
		return nil, err
	}

	outputs, err := p.model.GetOutputTensors()
	if err != nil {
		return nil, err
	}

	rawOutput := outputs[0].Data().([]float32)
	softmaxOutput := softmax(rawOutput)

	return softmaxOutput, nil
}
