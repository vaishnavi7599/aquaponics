// Function to get all input values from the form
function getAllInputValues() {
    // Get location value
    const location = document.getElementById('selected').value;
    
    // Get all input fields
    const inputs = document.querySelectorAll('.entries input[type="number"]');
    
    // Create an array with location as the first element
    const values = [parseInt(location)];
    
    // Add each input value to the array
    inputs.forEach(input => {
        // Use parseFloat to convert the input value to a number
        // If input is empty, use 0 as default
        values.push(parseFloat(input.value) || 0);
    });
    
    return values;
}

// Function to predict using the ML model
async function predict() {
    try {
        // Get all input values
        const inputValues = getAllInputValues();
        
        // Check if location is selected
        if (!inputValues[0] && inputValues[0] !== 0) {
            document.getElementById('res').textContent = "Please select a location!";
            document.getElementById('resm').textContent = "";
            return;
        }
        
        // Show loading message
        document.getElementById('res').textContent = "Calculating prediction...";
        document.getElementById('resm').textContent = "";
        
        // Send data to Python backend (updated endpoint)
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: inputValues }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        // Get prediction result
        const data = await response.json();
        
        // Display prediction result
        document.getElementById('res').textContent = data.result;
        
        // Optional: Display a message about the accuracy
        document.getElementById('resm').textContent = "Prediction based on Random Forest model (R² score: 0.9942)";
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('res').textContent = "An error occurred during prediction.";
        document.getElementById('resm').textContent = "Please check the console for details.";
    }
}

// Function to handle location selection
function myFunction() {
    // You can add any functionality needed when location changes
    console.log("Location selected:", document.getElementById('selected').value);
}