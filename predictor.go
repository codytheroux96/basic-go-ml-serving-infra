package main

import (
	"image"
	"image/color"
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

// convert grayscale image to float32 array
func imageToFloat32Array(img image.Image) []float32 {
	bounds := img.Bounds()
	result := make([]float32, 0, bounds.Dx()*bounds.Dy())

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			gray := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			result = append(result, float32(gray.Y)/255.0)
		}
	}
	return result
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
