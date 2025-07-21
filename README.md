# knowledgePants

By [<b>Mohammed khathir</b>](https://khathir.vercel.app)

## About The Project


The Document Question Answering System is a sophisticated tool designed to streamline information retrieval from vast document collections. Built on a foundation of advanced natural language processing techniques. Leveraging the LangChain framework and Google Generative AI, it ingests documents, converts them into vector embeddings, and employs the Retrieval augmentation generation(RAG) architecture for accurate question answering. Users can input queries through the intuitive interface, with the system retrieving precise answers based on the document context. With its efficiency, accuracy, and scalability, the system finds applications in research, knowledge management, education, and customer support, representing a significant advancement in information access technology.

## Library Requirements

- **FastAPI** – Backend web framework
- **Pinecone** – Vector database for similarity search
- **Google Gemini & Groq** – LLMs for response generation
- **PyMuPDF** – PDF text extraction
- **uuid / datetime** – Session management
- **dotenv** – Environment variable handling
- **Uvicorn** – ASGI server

## Getting Started

This will help you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Installation Steps

### Option 1: Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/khathirDev/knowledgePants-backend.git
     cd knowledgepants-backend
     ```

2. **Create a Virtual Environment** (Optional but recommended)
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     # Windows
      python -m venv venv
      

     # macOS/Linux
      python3 -m venv venv
      
     ```

3. **Activate the Virtual Environment** (Optional)
   - Activate the virtual environment based on your operating system:
       ```
       # Windows
       venv\Scripts\activate

       # macOS/Linux
       source venv/bin/activate
       ```

4. **Install Dependencies**
   
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

5. **Run the Project**
   - Start the project by running the appropriate command.
     ```
    uvicorn app:app --reload

     ```

6. **Access the Project**
   - Open a web browser or the appropriate client to access the project.


## API Key Setup

To use this project, you need an API key from Google Gemini Large Language Model and Groq. Follow these steps to obtain and set up your API key:

1. **Get API Key:**
   - Visit the Provided Links [Groq API](https://console.groq.com/keys) and [Google API](https://aistudio.google.com/app/apikey).
   - Follow the instructions to create an account and obtain your API key.

2. **Set Up API Key:**
   - Create a file named `.env` in the project root.
   - Add your API key to the `.env` file:
     ```dotenv
     API_KEY=your_api_key_here
     ```

   **Note:** Keep your API key confidential. Do not share it publicly or expose it in your code.<br>


## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

• **Report bugs**: If you encounter any bugs, please let us know. Open up an issue and let us know the problem.

• **Contribute code**: If you are a developer and want to contribute, follow the instructions below to get started!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

• **Suggestions**: If you don't want to code but have some awesome ideas, open up an issue explaining some updates or improvements you would like to see!

#### Don't forget to give the project a star! Thanks again!

