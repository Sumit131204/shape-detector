import React from "react";

const ImageDisplay = ({ imageUrl, title, hidden = false }) => {
  if (hidden) return null;

  return (
    <div className="d-flex flex-column align-items-center justify-content-center w-100">
      <div className="image-display-container" style={{ maxWidth: "400px" }}>
        <h5 className="mb-3 text-center">{title}</h5>
        <div className="text-center">
          <img
            src={imageUrl}
            alt={title}
            className="img-fluid rounded"
            style={{
              maxHeight: "300px",
              maxWidth: "100%",
              objectFit: "contain",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default ImageDisplay;
