package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	randomforest "github.com/malaschitz/randomForest"
	metrics "github.com/pa-m/sklearn/metrics"
	model_selection "github.com/pa-m/sklearn/model_selection"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
)

func maintest() {
    // Load CSV file
    file, err := os.Open("bq-results-20240122-104014-1705924340703 (1).csv")
    if err != nil {
        fmt.Println("Error opening file:", err)
        return
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()

    if err != nil {
        fmt.Println("Error reading CSV:", err)
        return
    }

    // Detect categorical columns and apply label encoding
    // var labelEncoders []*preprocessing.LabelEncoder
    var X [][]float64
    var y []int
	
	fmt.Println(len(records[0][:]), len(records[:][0]), records[0][1], records[1][0])
	fmt.Println(records[0][0], records[0][1], records[0][2], records[1][:])
	fmt.Println(records[0][0], records[1][0], records[2][0])
	for idx, _ := range records[0] {
		isCategorical := false
		for _, record := range records[1][:] {
			_, err := strconv.ParseFloat(record, 64)
			if err != nil {
				isCategorical = true
				fmt.Println(isCategorical)
				// break
			}
			fmt.Println(isCategorical, record, idx)
		}
	}
    // for i, _ := range records[0] {
    //     isCategorical := false
    //     for _, record := range records[1:] {
    //         _, err := strconv.ParseFloat(record[i], 64)
    //         if err != nil {
    //             isCategorical = true
    //             break
    //         }
    //     }
    //     if isCategorical {
    //         encoder := preprocessing.NewLabelEncoder()
    //         var encodedColumn []int
    //         for _, record := range records[1:] {
    //             encodedValue := encoder.FitTransform([]string{record[i]})[0]
    //             encodedColumn = append(encodedColumn, encodedValue)
    //         }
    //         labelEncoders = append(labelEncoders, encoder)
    //         X = append(X, []float64{float64(encodedValue)})
    //     } else {
    //         var numericColumn []float64
    //         for _, record := range records[1:] {
    //             floatValue, _ := strconv.ParseFloat(record[i], 64)
    //             numericColumn = append(numericColumn, floatValue)
    //         }
    //         X = append(X, numericColumn)
    //     }
    // }

    // Prepare target vector (y)
    // for _, record := range records[1:] {
    //     classValue, err := strconv.Atoi(record[len(record)-1])
    //     if err != nil {
    //         fmt.Println("Error converting class value to int:", err)
    //         return
    //     }
    //     y = append(y, classValue)
    // }

    // Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test := model_selection.TrainTestSplit(ConvertToDense(X), ConvertToIntMatrix(y), 0.2, 42)

    // Preprocess data (standardization)
    scaler := preprocessing.NewStandardScaler()
    scaler.Fit(X_train, y_train)
    X_train, y_train = scaler.FitTransform(X_train, y_train)
    X_test, y_test = scaler.Transform(X_test, y_test)

	y_train_slice := y_train.RawMatrix().Data
	y_train_int := make([]int, len(y_train_slice))
	for i, v := range y_train_slice {
		y_train_int[i] = int(v)
	}

    // Train Random Forest model
    forest := randomforest.Forest{}		
	forest.Data = randomforest.ForestData{X: ConvertToSlice(X_train), Class: y_train_int}
	forest.Train(1000)

	var predictions [][]float64
    numRows, _ := X_test.Dims()
    for i := 0; i < numRows; i++ {
        // Get a single row from X_test
        var row []float64
        for j := 0; j < X_test.RawMatrix().Cols; j++ {
            v := X_test.At(i, j)
            row = append(row, v)
        }

        // Make prediction for the row and append to predictions
        prediction := forest.Vote(row)
        predictions = append(predictions, prediction)
    }

	var nilDense *mat.Dense
	normalize, sampleWeight := true, nilDense
	// Evaluate model if needed
	fmt.Println(metrics.AccuracyScore(y_test, ConvertToDense(predictions), normalize, sampleWeight))
}

func ConvertToDense(data [][]float64) *mat.Dense {
	rows := len(data)
	cols := len(data[0])

	matData := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matData[i*cols+j] = data[i][j]
		}
	}

	return mat.NewDense(rows, cols, matData)
}

func ConvertToIntMatrix(slice []int) *mat.Dense {
	// Create a new dense matrix with one row and the length of the slice
	rows, cols := 1, len(slice)
	data := make([]float64, len(slice))

	// Populate the matrix with the elements of the slice
	for i, v := range slice {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

func ConvertToSlice(m *mat.Dense) [][]float64 {
	rows, cols := m.Dims()
	data := m.RawMatrix().Data

	// Convert 1D slice to 2D slice
	result := make([][]float64, rows)
	idx := 0
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = data[idx]
			idx++
		}
	}

	return result
}


