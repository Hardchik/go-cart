package labelencoder

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// LabelEncoder struct to hold the mapping
type LabelEncoder struct {
    Mapping map[string]int `json:"mapping"`
}

// NewLabelEncoder creates a new instance of LabelEncoder
func NewLabelEncoder() *LabelEncoder {
    return &LabelEncoder{
        Mapping: make(map[string]int),
    }
}

// Fit method to fit the encoder to a list of strings
func (le *LabelEncoder) Fit(labels []string) {
    // Use a map to track unique labels
    uniqueLabels := make(map[string]struct{})

    // Collect unique labels
    for _, label := range labels {
        uniqueLabels[label] = struct{}{}
    }

    // Sort and assign integers starting from 1
    sortedUniqueLabels := make([]string, 0, len(uniqueLabels))
    for label := range uniqueLabels {
        sortedUniqueLabels = append(sortedUniqueLabels, label)
    }
    sort.Strings(sortedUniqueLabels)

    // Assign integers starting from 1
    for i, label := range sortedUniqueLabels {
        le.Mapping[label] = i + 1
    }
}
// Transform method to encode a single label
func (le *LabelEncoder) Transform(label string) int {
    if val, ok := le.Mapping[label]; ok {
        return val
    }
    // If label not found, return -1 or any default value as needed
    return -1
}

// Encode method to encode a list of labels
func (le *LabelEncoder) Encode(labels []string) []int {
    encoded := make([]int, len(labels))
    for i, label := range labels {
        encoded[i] = le.Transform(label)
    }
    return encoded
}

// Save method to save the encoder mapping to a file
func (le *LabelEncoder) Save(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    return encoder.Encode(le)
}

// LoadLabelEncoder loads the encoder mapping from a file
func LoadLabelEncoder(filename string) (*LabelEncoder, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var le LabelEncoder
    decoder := json.NewDecoder(file)
    if err := decoder.Decode(&le); err != nil {
        return nil, err
    }
    return &le, nil
}

func LE() {
    // Example usage
    labels := []string{"banana", "apple", "orange", "banana", "apple", "apple"}
    encoder := NewLabelEncoder()
    encoder.Fit(labels)

    // Save encoder mapping to a file
    if err := encoder.Save("encoder.json"); err != nil {
        fmt.Println("Error saving encoder:", err)
        return
    }
    fmt.Println("Encoder saved successfully")

    // Load encoder mapping from file
    loadedEncoder, err := LoadLabelEncoder("encoder.json")
    if err != nil {
        fmt.Println("Error loading encoder:", err)
        return
    }

    // Use the loaded encoder
    encoded := loadedEncoder.Encode(labels)
    fmt.Println("Encoded labels using loaded encoder:", encoded)
}
