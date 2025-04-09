import React from "react";
import { Card, ListGroup, Badge } from "react-bootstrap";

const ShapeResult = ({ shapes }) => {
  // Function to get badge color based on shape type
  const getBadgeColor = (shape) => {
    const colorMap = {
      triangle: "warning",
      square: "primary",
      rectangle: "info",
      pentagon: "success",
      circle: "danger",
    };
    return colorMap[shape] || "secondary";
  };

  return (
    <Card>
      <Card.Header className="bg-light">
        <h4 className="mb-0">Detected Shapes</h4>
      </Card.Header>
      <Card.Body>
        {shapes.length === 0 ? (
          <p className="text-muted">No shapes detected in the image.</p>
        ) : (
          <div>
            <p>Found {shapes.length} shape(s) in the image:</p>
            <ListGroup variant="flush">
              {shapes.map((item, index) => (
                <ListGroup.Item
                  key={index}
                  className="d-flex justify-content-between align-items-center"
                >
                  <div>
                    <Badge bg={getBadgeColor(item.shape)} className="me-2">
                      {item.shape.charAt(0).toUpperCase() + item.shape.slice(1)}
                    </Badge>
                  </div>
                  <div className="text-muted">
                    Area: {Math.round(item.area)} pixels
                  </div>
                </ListGroup.Item>
              ))}
            </ListGroup>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ShapeResult;
