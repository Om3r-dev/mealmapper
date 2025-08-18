import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [items, setItems] = useState([]);
  const [data, setData] = useState({}); // Store full response data
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file before scanning.");
      return;
    }
    
    setError("");
    setLoading(true);
    setItems([]); // Clear previous results
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("https://https://smooth-forks-play.loca.lt//detect_foods", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Server error: " + res.statusText);
      }

      const data = await res.json();
      console.log("Full response:", data); // Debug logging
      console.log("Items array:", data.items); // Debug logging
      console.log("Items length:", data.items?.length); // Debug logging
      
      // Store full response data
      setData(data);
      
      // Ensure we're setting an array
      const itemsList = Array.isArray(data.items) ? data.items : [];
      setItems(itemsList);
      
    } catch (err) {
      console.error("Upload error:", err);
      setError("Upload failed: " + err.message);
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h1>MealMapper AI MVP</h1>
      <p>Take a photo of your fridge to detect ingredients. This is just an MVP (minimum viable product)</p>
      
      <input
        id="file"
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(e) => {
          setFile(e.target.files[0]);
          setError("");
          setItems([]); // Clear previous results when new file selected
        }}
        style={{ marginBottom: "10px" }}
      />
      <br />
      
      <div id="btn" style={{ margin: "10px 0" }}>
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? "Scanning..." : "Scan"}
        </button>
      </div>
      
      {error && <div style={{ color: "red", margin: "10px 0" }}>{error}</div>}
      
      {loading && <div style={{ color: "blue" }}>Processing image...</div>}
      
      {items.length > 0 && (
        <div>
          <h3>🎯 Detected Items ({items.length}):</h3>
          <div style={{ textAlign: "left", maxHeight: "400px", overflowY: "auto" }}>
            
            {/* High Confidence Items */}
            {data.high_confidence && data.high_confidence.length > 0 && (
              <div style={{ marginBottom: "15px" }}>
                <h4 style={{ color: "green", margin: "5px 0" }}>✅ High Confidence:</h4>
                <ul style={{ margin: "0 0 0 20px" }}>
                  {data.high_confidence.map((item, index) => (
                    <li key={`high-${index}`} style={{ marginBottom: "2px" }}>
                      <strong>{item.name}</strong> ({item.confidence}%)
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Medium Confidence Items */}
            {data.medium_confidence && data.medium_confidence.length > 0 && (
              <div style={{ marginBottom: "15px" }}>
                <h4 style={{ color: "orange", margin: "5px 0" }}>🤔 Medium Confidence:</h4>
                <ul style={{ margin: "0 0 0 20px" }}>
                  {data.medium_confidence.map((item, index) => (
                    <li key={`med-${index}`} style={{ marginBottom: "2px" }}>
                      {item.name} ({item.confidence}%)
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Low Confidence Items */}
            {data.low_confidence && data.low_confidence.length > 0 && (
              <div style={{ marginBottom: "15px" }}>
                <h4 style={{ color: "gray", margin: "5px 0" }}>❓ Possible Items:</h4>
                <ul style={{ margin: "0 0 0 20px" }}>
                  {data.low_confidence.slice(0, 20).map((item, index) => (
                    <li key={`low-${index}`} style={{ marginBottom: "2px", fontSize: "14px" }}>
                      {item.name} ({item.confidence}%)
                    </li>
                  ))}
                </ul>
                {data.low_confidence.length > 20 && (
                  <p style={{ fontSize: "12px", color: "gray" }}>
                    ...and {data.low_confidence.length - 20} more possible items
                  </p>
                )}
              </div>
            )}
            
          </div>
          
          <div style={{ fontSize: "12px", color: "gray", marginTop: "10px" }}>
            Total detected: {data.total_detected || items.length} items from {data.total_concepts} concepts analyzed
          </div>
        </div>
      )}
      
      {!loading && !error && items.length === 0 && file && (
        <div style={{ color: "gray" }}>No items detected or scan not performed yet.</div>
      )}
    </div>
  );
}

export default App;