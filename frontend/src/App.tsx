import React, { useEffect } from 'react';
import { Container, Row, Col, Button, Form } from 'react-bootstrap';
import './App.css';

// Declare the particlesJS type
declare global {
  interface Window {
    particlesJS: any;
  }
}

const App: React.FC = () => {
  useEffect(() => {
    // eslint-disable-next-line
    window.particlesJS.load('particles-js', 'assets/particles.json', function() {
      console.log('callback - particles.js config loaded');
    });
  }, []);

  return (
    <div>
      <div id="particles-js"></div>
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
    </div>
  );
}

export default App;