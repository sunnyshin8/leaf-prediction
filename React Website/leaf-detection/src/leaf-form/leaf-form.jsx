
import React, { useState } from 'react';
import axios from 'axios';

function LeafDetectionForm() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [responseText, setResponseText] = useState('');

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleSubmit = async () => {
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      const response = await axios.post('https://leaf-detection.onrender.com/upload', formData);
  
      // Handle the response, e.g., update responseText state variable
      setResponseText(response.data);
    } catch (error) {
      console.error('Error:', error);
      // Handle the error, e.g., update responseText state variable
      setResponseText('An error occurred during image upload.');
    }
  };
  

  return (
    <div className="form-container">
      <input type="file" onChange={handleImageChange} />
      <button onClick={handleSubmit}>Submit</button>
      <p className="response-text">{responseText}</p>
    </div>
  );
}

export default LeafDetectionForm;
