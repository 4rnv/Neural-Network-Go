package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Network struct {
	inputs         int
	hiddens        int
	outputs        int
	hidden_weights *mat.Dense
	output_weights *mat.Dense
	learning_rate  float64
}

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func add_scalar(scalar_value float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	array := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		array[x] = scalar_value
	}
	n := mat.NewDense(r, c, array)
	return add(m, n)
}

func input(r int, c int) []float64 {
	var filled_array = make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		fmt.Printf("Enter element %d: ", i+1)
		fmt.Scanf("%f", &filled_array[i])
	}
	return filled_array
}

func format(output mat.Matrix) {
	fmt.Printf("%.2f", mat.Formatted(output, mat.Prefix("    "), mat.FormatPython()))
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoid_prime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

func (net *Network) train(inputData []float64, target_data []float64) {

	// forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hidden_inputs := dot(net.hidden_weights, inputs)
	hidden_outputs := apply(sigmoid, hidden_inputs)
	final_inputs := dot(net.output_weights, hidden_outputs)
	final_outputs := apply(sigmoid, final_inputs)

	// find errors
	targets := mat.NewDense(len(target_data), 1, target_data)
	output_errors := subtract(targets, final_outputs)
	hidden_errors := dot(net.output_weights.T(), output_errors)

	// backpropagate
	net.output_weights = add(net.output_weights,
		scale(net.learning_rate,
			dot(multiply(output_errors, sigmoid_prime(final_outputs)),
				hidden_outputs.T()))).(*mat.Dense)

	net.hidden_weights = add(net.hidden_weights,
		scale(net.learning_rate,
			dot(multiply(hidden_errors, sigmoid_prime(hidden_outputs)),
				inputs.T()))).(*mat.Dense)
}

// The part (net Network) is called the method receiver. It defines the type that the method is associated with. In this case, net is the receiver variable of type Network. By defining a method with a receiver, you can access the fields and methods of the Network instance within the method.
func (net Network) predict(input_data []float64) mat.Matrix {

	// forward propagation: input_data goes into matrix inputs
	inputs := mat.NewDense(len(input_data), 1, input_data)

	// standard multiplication being performed on inputs matrix with hidden_weights in net struct
	hidden_inputs := dot(net.hidden_weights, inputs)
	hidden_outputs := apply(sigmoid, hidden_inputs)
	final_inputs := dot(net.output_weights, hidden_outputs)
	final_outputs := apply(sigmoid, final_inputs)
	//format(final_outputs)
	return final_outputs
}

func random_array(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	// fmt.Println("Uniform Distribution: ", dist)
	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	// fmt.Println("Random Array: ", data)
	return
}

// Creating an instance/object of network struct.
// Why are multiple words to describe the same thing? Object/Instance, Class/Struct, Method/Function
func create_network(input, hidden, output int, rate float64) (net Network) {
	net = Network{inputs: input, hiddens: hidden, outputs: output, learning_rate: rate}
	net.hidden_weights = mat.NewDense(net.hiddens, net.inputs, random_array(net.inputs*net.hiddens, float64(net.inputs)))
	net.output_weights = mat.NewDense(net.outputs, net.hiddens, random_array(net.hiddens*net.outputs, float64(net.hiddens)))
	fmt.Println("Network Created: ")
	fmt.Println("Hidden Weights: ")
	//format(net.hidden_weights)
	fmt.Println("Output Weights: ")
	//format(net.output_weights)
	return
}

func save(net Network) {
	h, err := os.Create("hweights.model")
	if err == nil {
		net.hidden_weights.MarshalBinaryTo(h)
	}
	defer h.Close()
	if err != nil {
		fmt.Println("Error saving model", err)
	}
	o, err := os.Create("oweights.model")
	if err == nil {
		net.output_weights.MarshalBinaryTo(o)
	}
	defer o.Close()
	if err != nil {
		fmt.Println("Error saving model", err)
	}
}

func load(net *Network) {
	h, err := os.Open("hweights.model")
	if err == nil {
		net.hidden_weights.Reset()
		net.hidden_weights.UnmarshalBinaryFrom(h)
	}
	defer h.Close()
	o, err := os.Open("oweights.model")
	if err == nil {
		net.output_weights.Reset()
		net.output_weights.UnmarshalBinaryFrom(o)
	}
	defer o.Close()
	//return
}

func mnist_train(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	for epochs := 0; epochs < 5; epochs++ {
		test_file, _ := os.Open("data/mnist_train.csv")
		r := csv.NewReader(bufio.NewReader(test_file))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			//fmt.Println(record)
			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			net.train(inputs, targets)
		}
		test_file.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}

func mnist_predict(net *Network) {
	t1 := time.Now()
	check_file, _ := os.Open("data/mnist_test.csv")
	defer check_file.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(check_file))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		outputs := net.predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
	accuracy := float64(score) / 10000.0
	fmt.Println("Accuracy: ", accuracy)
	log_file, err := os.OpenFile("logs.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening log file:", err)
		return
	}
	defer log_file.Close()

	writer := bufio.NewWriter(log_file)
	log_entry := fmt.Sprintf("Network: (%d,%d,%d,%f) ", net.inputs, net.hiddens, net.outputs, net.learning_rate) + fmt.Sprintf("Time taken: %s, Score: %d, Accuracy: %.4f\n", elapsed, score, accuracy)
	writer.WriteString(log_entry)
	writer.Flush()
}

func data_from_image(file_path string) (pixels []float64) {

	img_file, err := os.Open(file_path)
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	defer img_file.Close()

	img, err := png.Decode(img_file)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}

	pixels = make([]float64, len(gray.Pix))
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.99) + 0.01
	}
	return
}

func predict_from_image(net Network, path string) (int, int) {
	input := data_from_image(path)
	output := net.predict(input)
	format(output)

	best := -1
	second_best := -1
	highest := 0.0
	second_highest := 0.0

	for i := 0; i < net.outputs; i++ {
		probability := output.At(i, 0)
		if probability > highest {
			second_best = best
			second_highest = highest

			best = i
			highest = probability
		} else if probability > second_highest {
			second_best = i
			second_highest = probability
		}
	}

	return best, second_best
}

func main() {
	net := create_network(784, 200, 10, 0.1)
	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	switch *mnist {
	case "train":
		mnist_train(&net)
		save(net)
	case "predict":
		load(&net)
		mnist_predict(&net)
	}
	if *file != "" {
		load(&net)
		best, second_best := predict_from_image(net, *file)
		fmt.Println("\nPrediction:", best, second_best)
	}
}
