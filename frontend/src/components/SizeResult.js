import React, { useState } from "react";
import { Card, Table } from "react-bootstrap";

const SizeResult = ({ measurements }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      style={{
        perspective: "1000px",
        marginBottom: "20px",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Card
        style={{
          transform: isHovered
            ? "rotateY(-5deg) rotateX(5deg)"
            : "rotateY(0deg) rotateX(0deg)",
          transition: "transform 0.3s ease, box-shadow 0.3s ease",
          transformStyle: "preserve-3d",
          boxShadow: isHovered
            ? "rgba(0, 0, 0, 0.1) 5px 5px 15px, rgba(0, 0, 0, 0.07) 15px 15px 20px"
            : "rgba(0, 0, 0, 0.1) 0px 4px 12px, rgba(0, 0, 0, 0.05) 0px 1px 3px",
          borderRadius: "10px",
          border: "1px solid rgba(255,255,255,0.2)",
          overflow: "hidden",
        }}
      >
        <Card.Header
          style={{
            backgroundColor: "var(--light-blue)",
            color: "var(--primary-black)",
            borderBottom: "1px solid rgba(0,0,0,0.05)",
            position: "relative",
            zIndex: 1,
            transform: isHovered ? "translateZ(10px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          <h4 className="mb-0">Size Measurements</h4>
        </Card.Header>
        <Card.Body
          style={{
            position: "relative",
            zIndex: 0,
            transform: isHovered ? "translateZ(5px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          {measurements.length === 0 ? (
            <p className="text-muted">No measurements available.</p>
          ) : (
            <div>
              <p>Detected {measurements.length} object(s) in the image:</p>
              <Table
                striped
                bordered
                hover
                responsive
                style={{
                  transform: isHovered ? "translateZ(8px)" : "translateZ(0)",
                  transition: "transform 0.3s ease",
                }}
              >
                <thead>
                  <tr
                    style={{
                      backgroundColor: "var(--light-blue)",
                      color: "var(--primary-black)",
                    }}
                  >
                    <th>#</th>
                    <th>Length (mm)</th>
                    <th>Breadth (mm)</th>
                    <th>Area (mm²)</th>
                  </tr>
                </thead>
                <tbody>
                  {measurements.map((item, index) => (
                    <tr
                      key={index}
                      style={{
                        transform: isHovered
                          ? `translateZ(${5 - index * 0.5}px)`
                          : "translateZ(0)",
                        transition: "transform 0.3s ease",
                      }}
                    >
                      <td>{index + 1}</td>
                      <td>{item.length_mm.toFixed(2)}</td>
                      <td>{item.breadth_mm.toFixed(2)}</td>
                      <td>{item.area_mm2.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default SizeResult;
