import React from 'react';
import { Container, Row, Col, Button, Form } from 'react-bootstrap';
import './App.css';

const App: React.FC = () => {
  return (
    <Container fluid className="App">
      <Row className="justify-content-md-center">
        <Col md="auto">
          <h1 className="mt-5">DistLM</h1>
          <p>Ready to start training...</p>
          <Form>
            <Form.Group controlId="formFile" className="mb-3">
              <Form.Label>Upload Model</Form.Label>
              <Form.Control type="file" />
            </Form.Group>
            <Form.Group controlId="formFileMultiple" className="mb-3">
              <Form.Label>Upload Data</Form.Label>
              <Form.Control type="file" multiple />
            </Form.Group>
            <Button variant="primary" type="submit">
              Start Training
            </Button>
          </Form>
        </Col>
      </Row>
    </Container>
  );
}

export default App;
