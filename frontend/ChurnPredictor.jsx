// ChurnPredictor.jsx
import React, { useState } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function ChurnPredictor() {
  const [customer, setCustomer] = useState({
    Tenure: 12,
    NumberOfAddress: 2,
    CashbackAmount: 100,
    DaySinceLastOrder: 7,
    OrderCount: 5,
    SatisfactionScore: 4.0
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const predictChurn = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, customer);
      setPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed');
    }
    setLoading(false);
  };

  return (
    <div className="churn-predictor">
      <h2>Customer Churn Prediction</h2>
      
      <div className="input-form">
        <input type="number" placeholder="Tenure" value={customer.Tenure} 
               onChange={e => setCustomer({...customer, Tenure: e.target.value})} />
        {/* Add other inputs similarly */}
      </div>

      <button onClick={predictChurn} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Churn Risk'}
      </button>

      {prediction && (
        <div className={`prediction ${prediction.risk_level}`}>
          <h3>Risk: {prediction.risk_level.toUpperCase()}</h3>
          <p>Probability: {(prediction.churn_probability * 100).toFixed(1)}%</p>
          <p>Will churn: {prediction.will_churn ? 'Yes' : 'No'}</p>
        </div>
      )}
    </div>
  );
}