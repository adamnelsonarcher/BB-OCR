We are testing the use of a Vision-Language Model (VLM), such as BLIP-2 or LLaVA, to extract metadata (title, author, publisher) from book cover images. The goal is to benchmark this approach against traditional OCR tools like Tesseract and EasyOCR, which have proven unreliable on stylized covers.

The test should involve:

1. **Model Setup**

   * Set up either BLIP-2 or LLaVA locally using a Conda environment or Docker.
   * Download the pre-trained model weights.
   * Ensure the model can accept image input and return text responses to specific prompts.

2. **Data Preparation**

   * Use a test dataset of 50–100 book cover images.
   * Organize these images in a folder and prepare a JSON or CSV file containing the expected metadata (ground truth) for comparison.

3. **Inference Script**

   * Write a Python script that:

     * Loads an image
     * Sends it through the VLM using prompts like:

       * "What is the title of this book?"
       * "Who is the author of this book?"
       * "Who is the publisher?"
     * Records the output and inference time
     * Stores everything in a JSON log per image

4. **Evaluation Script**

   * Create another script that compares model output to the ground truth metadata.
   * It should measure:

     * Accuracy (exact match or fuzzy match)
     * Average inference time per image

5. **Output**

   * Save a structured log (JSON or CSV) with fields like:

     * image name
     * VLM output
     * expected metadata
     * inference time
     * whether it matched or not
   * Optional: generate a table or chart comparing accuracy and speed vs. OCR

The main focus is to determine if a local VLM can outperform OCR in metadata extraction without incurring API costs. We are aiming for a fast, scalable solution that could process 800,000+ book covers locally. Start with just 1 model and 50 images.