package main

import (
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

func (p *Predictor) Predict(input []float32) ([]float32, error) {
	t := tensor.New(tensor.WithShape(1, 1, 28, 28), tensor.WithBacking(input))

	if err := p.model.SetInput(0, t); err != nil {
		return nil, err
	}

	if err := p.backend.Run(); err != nil {
		return nil, err
	}

	outputs := p.model.GetInputTensors()

	return outputs[0].Data().([]float32), nil
}
