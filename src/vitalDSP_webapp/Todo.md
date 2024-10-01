
## Feature List for the Web App (Dash + Plotly)

### 1. File Upload Interface:

- **Supported Formats**: Allow users to upload files in .csv, .edf, and .txt formats.
- **Upload Component**: Use Dash’s dcc.Upload() component for file uploads.
- **File Type Validation**: Ensure the correct file type is uploaded and provide error messages for unsupported formats.
- **Chunked Data Loading**:
    - Initialize chunk size before processing.
    - After file upload, only load and display a chunk of the data.
    - When the user interacts with the plot (e.g., dragging/zooming), load the next data chunk into the graph dynamically.

### 2. Dynamic Data Visualization with Plotly:

- **Plotly Graph Component**: Visualize signal data using dcc.Graph().
- **Real-Time Chunk Loading**: Use Plotly’s drag interaction (relayoutData or restyleData callbacks) to load the next data chunk when the user drags to the boundary of the current chunk.
- **Zoom and Pan Control**: Allow users to zoom and pan through data, dynamically loading the appropriate chunk in real-time.

### 3. Chunk Processing:

- **Locking Feature**:
    - Utilize Plotly’s drag-zoom tool to allow users to lock the start and end points of the data they want to process.
    - Once locked, only the selected data segment will be processed.
- **Display Lock Controls**: Add buttons or markers on the graph to allow users to reset or modify the lock points.

### 4. Processing Options (Dropdown or Sidebar):

- **Dropdown Menu**:
    - A dropdown menu (or sidebar) for processing options like filtering, artifact removal, and signal transformation.
- **Suggested UI Component**: Use dcc.Dropdown() or a dcc.Slider() for filtering parameters.
- **Filtering Options**: Include options for different filters (e.g., low-pass, high-pass, bandpass filters).
- **Artifact Removal**: Provide options for removing common signal artifacts (e.g., noise).
- **Transformation**: Allow users to apply transformations like wavelet transforms, FFT, etc.

### 5. Signal Quality Index (SQI):

- **Compute SQI**:
    - Allow users to compute the signal quality index (SQI) of the selected part.
    - Allow for the computation of sub-segments within the locked area (define sub-segment duration).
- **Display SQI on Hover**: When hovering over a segment of the plot, display the computed SQI score using Plotly’s hover event callback (hoverData).

### 6. Visualization Enhancements:

- **Graph Annotations**: Highlight the selected chunk and locked region on the graph.
- **Color-Coded SQI**: Visually color-code different parts of the signal based on the SQI score (e.g., green for good quality, red for low quality).

### 7. User-Friendly Layout:

- **Responsive Design**: Make the app layout responsive so it works on different screen sizes.
- **Interactive Help**: Provide tooltips or help icons to explain features like chunk processing, filtering, etc.