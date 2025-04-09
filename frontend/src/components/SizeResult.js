import React from "react";
import { Card, Table } from "react-bootstrap";

const SizeResult = ({ measurements }) => {
  return (
    <Card>
      <Card.Header className="bg-light">
        <h4 className="mb-0">Size Measurements</h4>
      </Card.Header>
      <Card.Body>
        {measurements.length === 0 ? (
          <p className="text-muted">No measurements available.</p>
        ) : (
          <div>
            <p>Detected {measurements.length} object(s) in the image:</p>
            <Table striped bordered hover responsive>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Length (mm)</th>
                  <th>Breadth (mm)</th>
                  <th>Area (mm²)</th>
                </tr>
              </thead>
              <tbody>
                {measurements.map((item, index) => (
                  <tr key={index}>
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
  );
};

export default SizeResult;
