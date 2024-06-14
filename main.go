package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"model_traning/labelencoder"
	"os"
	"strconv"

	dataframe "github.com/rocketlaunchr/dataframe-go"
	base "github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
)

func loadCSV(filePath string) ([][]string, error) {
	// Open the CSV file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Create a new CSV reader
	reader := csv.NewReader(file)

	// Read all CSV data
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records, nil
}



func main() {
	// Load CSV file
	data, err := loadCSV("data_for_last_mile.csv")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Extract headers
	headers := data[0]
	data = data[1:] // Remove header row from data
    // fmt.Println(headers)
	// Initialize LabelEncoder for columns with string values
	encoders := make([]*labelencoder.LabelEncoder, len(headers))

	// Convert string data to int using LabelEncoder for selected columns
	var encodedData [][]interface{}
	for col := 0; col < len(headers); col++ {
		var colValues []interface{}
		isString := false
		for _, row := range data {
			_, errFloat := strconv.ParseFloat(row[col], 64)
			_, errInt := strconv.ParseInt(row[col], 10, 64)
            if errFloat != nil && errInt != nil {
                isString = true
            }
			colValues = append(colValues, row[col])
		}
		if isString {
            fmt.Println(headers[col])
			values := InterfaceToString(colValues)
			// Initialize a new LabelEncoder for the column
			encoders[col] = labelencoder.NewLabelEncoder()
			encoders[col].Fit(values)
			encodedCol := encoders[col].Encode(values)
			// Fit and transform column data
			// encodedCol, _ := encoders[col].FitTransform(nil, NewStringMatrix(len(colValues), 1, colValues))
			// Convert encoded column to []int
			numRows := len(encodedCol)
			colData := make([]interface{}, numRows)
			for i := 0; i < numRows; i++ {
				colData[i] = encodedCol[i]
			}
			// fmt.Println(colData)
			encodedData = append(encodedData, colData)
		} else {
			encodedData = append(encodedData, colValues)
		}
	}

	// Print headers
	// fmt.Println("Column Headers:")
	// fmt.Println(headers)

	// fmt.Println("First 5 rows after label encoding:")
	transposedData := transpose(encodedData)
	// fmt.Println("Column Headers:")
	// fmt.Println(headers)
	// fmt.Println(transposedData[0:5])

	// Create a new DataFrame
	series := make([]dataframe.Series, len(headers))
    for i, header := range headers {
        var s dataframe.Series
        // Determine the type of the series based on the type of the first value
        switch transposedData[0][i].(type) {
        case string:
            s = dataframe.NewSeriesFloat64(header, nil)
        case int:
            s = dataframe.NewSeriesInt64(header, nil)
        case float64:
            s = dataframe.NewSeriesFloat64(header, nil)
        default:
            panic(fmt.Sprintf("Unsupported data type for column %s", header))
        }
        series[i] = s
    }
	fmt.Println(series)
	// Add data to the series
    for _, row := range transposedData {
        for i, value := range row {
            series[i].Append(value)
        }
    }

    // Create a new DataFrame
    df := dataframe.NewDataFrame(series...)

	// Print the DataFrame
	fmt.Println(df)
	mlData := base.ConvertDataFrameToInstances(df, 7)
	shuffleData := base.Shuffle(mlData)
	train, test := base.InstancesTrainTestSplit(shuffleData, 0.2)
	rf := ensemble.NewRandomForest(100, 100)
	rf.String()
	err = rf.Fit(train)
	if err != nil {
		log.Fatal(err)
	}

	// Predict on the testing set
	predictions, err := rf.Predict(test)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(train, test)
	fmt.Println(predictions)
	// rf.Save("random_forest_model")
	// Save label encoder models for columns with string values
	// for col, encoder := range encoders {
	// 	if encoder != nil {
	// 		err := encoder.Save(fmt.Sprintf("encoder_%s.model", headers[col]))
	// 		if err != nil {
	// 			fmt.Printf("Error saving encoder for column %s: %v\n", headers[col], err)
	// 		} else {
	// 			fmt.Printf("Label encoder for column %s saved successfully\n", headers[col])
	// 		}
	// 	}
	// }
}

func InterfaceToString(val []interface{}) []string {
    stringSlice := make([]string, len(val))
    for i, v := range val {
        stringSlice[i] = fmt.Sprintf("%v", v)
    }
    return stringSlice
}

func transpose(matrix [][]interface{}) [][]interface{} {
    if len(matrix) == 0 {
        return nil
    }

    numRows := len(matrix)
    numCols := len(matrix[0])

    transposed := make([][]interface{}, numCols)
    for j := 0; j < numCols; j++ {
        transposed[j] = make([]interface{}, numRows)
        for i := 0; i < numRows; i++ {
            transposed[j][i] = matrix[i][j]
        }
    }

    return transposed
}

func subsetMatrix(superset [][]interface{}, columnsToSelect []int) [][]interface{} {
    numRows := len(superset)
    subset := make([][]interface{}, numRows)

    for i := 0; i < numRows; i++ {
        subset[i] = make([]interface{}, len(columnsToSelect))
        for j, colIndex := range columnsToSelect {
            subset[i][j] = superset[i][colIndex]
        }
    }

    return subset
}
