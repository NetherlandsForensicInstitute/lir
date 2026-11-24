import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import validator from "@rjsf/validator-ajv8";
import Form from '@rjsf/mui';

import schema from '../lir.schema.json';

const log = (type) => console.log.bind(console, type);

// Start the form with a standard minimal configuration, so that the conditional sections (data, lrsystem)
// are visible from the beginning; the user can change the discriminators to explore the other options
const initialFormData = {
  experiments: [
    {
      strategy: "single_run",
      data: {
        setup: "split_data",
        provider: {
          method: "synthesized_normal_binary",
        },
        splits: {
          strategy: "train_test_instances",
        },
      },
      lrsystem: {
        architecture: "specific_source",
      },
    },
  ],
};

// the __comment__ fields only exist to show documentation in the form; remove them from the output
function stripComments(value) {
  if (Array.isArray(value)) {
    return value.map(stripComments);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => key !== "__comment__")
        .map(([key, val]) => [key, stripComments(val)])
    );
  }
  return value;
}

function App() {
  const [output, setOutput] = useState("");
  return (
    <div>
      <Form
        schema={schema}
        validator={validator}
        formData={initialFormData}
        onChange={log("changed")}
        onSubmit={(e) => {
          log("submitted");
          // JSON is a subset of YAML, so this output can be saved as a .yaml configuration file
          setOutput(JSON.stringify(stripComments(e.formData), null, 2));
        }}
        onError={(errors) => {
          log("errors");
          setOutput("");
        }}
      />
      {output && (
        <>
          <h3>Generated configuration</h3>
          <pre style={{ whiteSpace: "pre-wrap" }}>{output}</pre>
        </>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("lir-gui")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
