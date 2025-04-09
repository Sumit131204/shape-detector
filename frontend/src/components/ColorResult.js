import React from "react";
import { Card, Table } from "react-bootstrap";

const ColorResult = ({ colors }) => {
  return (
    <Card className="mb-4">
      <Card.Header className="bg-light">
        <h4 className="mb-0">Color Detection Results</h4>
      </Card.Header>
      <Card.Body>
        <Table striped bordered hover>
          <thead>
            <tr>
              <th>Shape</th>
              <th>Color</th>
              <th>Sample</th>
              <th>RGB Value</th>
            </tr>
          </thead>
          <tbody>
            {colors.map((color, index) => (
              <tr key={index}>
                <td>{color.shape}</td>
                <td>{color.color}</td>
                <td>
                  <div
                    style={{
                      backgroundColor: color.hex,
                      width: "30px",
                      height: "30px",
                      borderRadius: "4px",
                      border: "1px solid #ddd",
                      margin: "0 auto",
                    }}
                  />
                </td>
                <td>
                  <code>{`(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>

        {colors.length === 0 && (
          <div className="text-center text-muted">
            <p>No colors detected.</p>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ColorResult;
