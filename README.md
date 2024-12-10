# Title: Developing a Paper Review Assistant Leveraging LLM Technology: Powered by ChatGPT4o API

**AI+X: Deep Learning Course Team Assignment**

**Team members**

**Name1:** 허다, **Student Number:** 2024051112, **Department:**:실내건축디자인 , **Email:** [dxu321030@gmail.com](mailto\:dxu321030@gmail.com)

**Name2:** 안미선, **Student Number:** 2019001349, **Department:** , **Email:** [dksaltjs@hanyang.ac.kr](mailto\:dksaltjs@hanyang.ac.kr)

**Name3:** 황소웅, **Student Number:** 2021055569, **Department:** , **Email:** [huangshaoxiong326@gmail.com](mailto\:huangshaoxiong326@gmail.com)

**3.	Video address:**https://youtu.be/2ZnbesNDK-I

### **I. Proposal (Option 1)**

**(1) Motivation: Why are you doing this?**

The writing of a review paper requires reading a large number of academic papers and summarizing their content to outline the overall framework of related work, the specific technical implementation details, and the methods' details and limitations. Typically, a good review paper requires summarizing at least 50 academic papers, usually between 50 and 100. Such an enormous amount of reading and summarizing is both time-consuming and labor-intensive.

With the rise of large language models (LLMs) such as ChatGPT, researchers now have a powerful tool to process large amounts of literature quickly. For example, by November 30, 2023, a search for "ChatGPT" on PubMed returns over 1,400 articles with titles containing "ChatGPT," highlighting its rapid growth just one year after its launch. Currently, people are still exploring how large language models, such as ChatGPT, can be used to assist in writing review papers. Studies have already evaluated ChatGPT's potential in academic writing [3, 4, 5, 6, 7, 8, 9, 10], and discussions about its use as a generator of scientific review articles are ongoing [11, 12, 13].

However, many of these studies were conducted before the release of the more advanced GPT-4, which could make their findings outdated. Furthermore, there have been no studies utilizing the latest ChatGPT model (GPT-4o - the most recent development from OpenAI aimed at improving paper writing and code development) for review paper writing.

Hence, we plan to call the latest GPT-4o API and develop a paper review assistant based on a Python environment. The goal is to remove token limitations on input academic paper PDFs, allowing directed summarization of related content using a designated prompt. This assistant will help researchers write their review papers more efficiently. Additionally, we plan to evaluate our approach using metrics like Accuracy, Semantic Coherence, Compression Ratio, Time Efficiency, etc., and compare GPT-4o API's performance with the GPT web interface and other large language models (e.g., Google Gemini, Claude).

**(2) Differences between Using GPT Web Interface vs. Calling ChatGPT API for Academic Paper Reading and Writing**

**2.1 Web Interface vs. API for Long Document Summarization**

**Limitations of the Web Interface for Long Documents:**

The web version of ChatGPT attempts to intelligently extract key parts of the document, typically prioritizing titles, abstracts, and the initial pages, but may not read the entire document, especially if it's very long.

The response generated must conform to token limitations even if the whole document is uploaded.

**Advantages of API:**

API provides a higher token limit, particularly the 32k version of GPT-4o, which can handle more content, making it more flexible for analyzing long documents.

Manual segmentation of texts can be performed, allowing the assistant to read and analyze documents thoroughly.

**2.2 Comparison of Usage Scenarios**

**Web Interface for Quick Overview and Preliminary Reading:**

When you want to quickly understand the main content of a PDF, the file upload feature on the web interface is convenient as it automatically extracts and summarizes the content.

**API for In-Depth Analysis of Long Documents:**

If a complete and detailed analysis of a long PDF document is required, using the API is more suitable. By manually managing the input text and utilizing GPT-4's higher token limit, you can achieve a more detailed and thorough analysis.

If your goal is to have the assistant perform a complete analysis of a lengthy document, it is recommended to use the API, which allows for appropriate text processing and segmented input to leverage GPT-4's high token limit for the most comprehensive results.

**2.3 How Web Interface (ChatGPT, GPT-4o) Handles Long Documents**

**System Performance and Browsing Experience:**

**Memory and Computing Limitations:** The web version of GPT-4o, despite having the capability to call various auxiliary tools to read files, is still subject to system performance and computational constraints.

**Reading Ability of Uploaded Files:** The web version typically attempts to incrementally process uploaded PDF files. For extremely long documents, the model may only extract partial content, particularly front pages or certain structured sections like titles or chapter summaries. This reduces system load and improves user experience but also means the entire document may not be fully read.

**Maximum Token Limit:** Even the web version of GPT-4o has a token limit when processing input. This means that although the model has access to the full content of an uploaded PDF, the response it generates might be limited by the token count, which generally maxes out around 8k or 32k tokens (depending on the model version and configuration).

**Incremental Processing and Summarization:**
When handling long documents, the web-based assistant often provides preliminary results, such as an overview or summary, and may not immediately perform an in-depth analysis of the entire document. Users need multiple interactions to guide the model through different sections, especially for very long documents.

**2.4 Advantages of Using API**

**Higher Token Limit:**

**Maximum Token Limit is Higher:** When calling through the OpenAI API, certain GPT-4 configurations (such as the 32k token version) can process very long inputs. Therefore, if you can manually extract the PDF text and submit it as a prompt to the API, the API form of GPT-4o can accept a larger input compared to the web version. For the 32k token version, this means processing approximately 25,000 words of text (depending on the content).

In the API form, you can explicitly control the input content and compress it within the maximum token limit so that the model can fully analyze the entire document content.

**Control and Flexibility:**

**Manual Text Management:** Through the API, you have complete control over the text content submitted to the model. This means you can manually select the most critical parts and discard others or split them into manageable segments.

**Step-by-Step Processing of Long Text:** For particularly lengthy PDF documents, they can be split into multiple segments, uploaded to the API in batches, analyzed step-by-step, and the output results consolidated. This method is especially effective for documents exceeding the model's single-input limit.

### **II. Datasets**

- **Describing Your Dataset**

(1) For large language models, we use the latest GPT-4o by directly accessing it through an API. This approach does not require training a dataset for the model but instead relies on pre-trained capabilities, similar to how transfer learning works. By using the pre-trained model, we avoid the complexity of training a new system from scratch. While this method does not involve generating or using training data, the steps for applying and utilizing the GPT-4o API are outlined in the diagram below.

![image](https://github.com/user-attachments/assets/67fc3d70-2ef1-4ddf-93b6-30d06a2bb1c7)
![image](https://github.com/user-attachments/assets/12672e16-1f35-4d23-91e3-afc364b8dc03)
![image](https://github.com/user-attachments/assets/1181894d-e580-4672-9b7d-d3d8ba49cac0)

(2) To facilitate reading academic paper PDFs with preset prompts and summarizing content relevant to those prompts, we curated a collection of topic-specific academic papers for input into the GPT-4o model. This process required establishing a database of academic paper PDFs.**As an example, we focused on the topic of zebrafish-based foodborne virus culture techniques.** Using academic search engines such as Google Scholar, we compiled a dataset of 25 related academic papers to serve as input for the large language model. **The specific database of 25 academic papers on the topic of zebrafish-based foodborne virus culture techniques is illustrated in the diagram below.**

![image](https://github.com/user-attachments/assets/ffa3212b-a849-4e5d-a289-f8b82cbb2ae5)

### **III. Methodology**

#### Explaining Your Choice of Algorithms (Methods)

The project utilizes OpenAI's GPT model to automatically generate summaries of academic papers, helping users create concise and informative literature reviews. The following methods were employed:

**1. API Integration:**

The core functionality of the assistant revolves around OpenAI's GPT model, integrated through the OpenAI API. The reason for choosing GPT is due to its advanced natural language understanding capabilities, which are very suitable for efficiently extracting key points and summarizing complex academic content.

**2. PDF Processing:**

The project uses the pdfplumber library to extract text from uploaded PDF files. This method is chosen because it can reliably handle different formats of academic papers while maintaining the text's integrity, making effective summarization possible.

**3. Input Prompt Setup:**

After extracting text from the PDF, a specific prompt is set up, and GPT-4 is used to summarize the paper's content according to the focus points in the prompt. This prompt-setting method ensures that GPT summarizes the academic content according to the user's desired perspective, such as focusing on methods, results, or key contributions.

**4. Text Summarization Logic:**

The extracted text is sent to the OpenAI API for summarization. The API processes the text and generates a summary version that includes the main contributions, research methods, and results. The text is first limited to a certain character count to comply with API constraints, ensuring that only relevant content is processed.**As detailed in the project's Python file:** 
**https://github.com/XuDuoXuDuoXuDuo/AI-X-Deep-Learning_Team-Project/blob/main/a%20Paper%20Review%20Assistant%20Leveraging%20LLM%20Technology%20Powered%20by%20ChatGPT4o%20API.ipynb**

#### Explaining Features

**Paper Review Assistant Functions and Code Flow Explanation:**

**3.1 Environment Setup:** The code begins by installing the necessary packages (openai and pdfplumber) and importing the required modules.As shown in the diagram below:

![image](https://github.com/user-attachments/assets/59b4aded-c759-469b-a6ae-a5222e003170)

**3.2 API Key Setup:** Set the OpenAI API key to enable integration with the model.As shown in the diagram below:

![image](https://github.com/user-attachments/assets/806fd5b6-7dae-4ee7-b63a-8c31ac58f89f)

**3.3 File Upload:** Use Google Colab's file upload feature to prompt users to manually upload PDF files.As shown in the diagram below:

![image](https://github.com/user-attachments/assets/b0ea664a-d2c6-4f4a-ba75-03f7e8f87621)

**3.4 Text Extraction:** Use the extract\_text\_from\_pdfs() function to iterate over uploaded PDFs and extract text while limiting the character count to ensure efficiency.As shown in the diagram below:

![image](https://github.com/user-attachments/assets/485a2cfb-b4b1-45bc-af2b-d53ecb016ab2)
![image](https://github.com/user-attachments/assets/8e24c6e6-563d-4196-bb60-e29e152b3868)
![image](https://github.com/user-attachments/assets/cc8c5a39-0e68-41ee-be95-c2ccb557137f)

**3.5 Input Prompt Setup:** After extracting the text, set up a prompt to further generate a focused summary of the text.As shown in the diagram below:

![image](https://github.com/user-attachments/assets/74fa42cd-6a73-4886-8d2f-a8427024c22b)

**3.6 Summary Invocation:** Use the extracted text to call the OpenAI API, and the API returns a summary of the paper.As shown in the diagram below:

![image](https://github.com/user-attachments/assets/38964a4b-bb9a-4fe5-91b1-b1f328f8f362)

**IV. Results:**

**4.1 **Using the 21-page academic paper titled "A robust human norovirus replication model in zebrafish larvae" as an example, we successfully performed a full-text reading of the PDF document in 1.07 seconds by utilizing the OpenAI GPT-4o API Key. Based on the preset prompt, the model generated a concise summary of the key points. The specific prompt and the model's summary response are as follows:

![image](https://github.com/user-attachments/assets/71947f53-d8cc-47b3-a899-8d10f79bbf27)

**4.2** The response generated by the GPT-4o API is as follows. **From the results, it is evident that the response is highly accurate. The model successfully performs a comprehensive reading and understanding of the academic paper PDF and provides precise answers to the key points based on the specified prompts.**

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. **Background and Advantages of Using Zebrafish as a Model:**
   - Zebrafish are optically transparent and have genetic, physiological, and pharmacological similarities to humans. They are cost-effective and compliant with ethical research principles, making them a valuable model for studying viral infections.
   - Zebrafish have a comparable immune system to humans and can be used to study host-pathogen interactions. They are widely used in research due to their high fecundity and low maintenance costs compared to other vertebrate models like rodents.
2. **Classification and Impact of Foodborne Viruses on Global Food Safety:**
   - Human noroviruses (HuNoVs) are the leading cause of viral gastroenteritis worldwide, resulting in significant morbidity and mortality. They are responsible for large outbreaks and pose a burden on healthcare systems.
   - The lack of robust small animal models has historically hindered the understanding and treatment of norovirus infections.
3. **Technical Details of Viral Culture in Zebrafish, Including Infection Methods:**
   - Zebrafish larvae were injected with a suspension of HuNoV positive stool samples into the yolk sac. This method allows for the study of viral replication within a living organism.
   - The method demonstrated successful replication of HuNoV, with high viral titers detectable for several days post-infection.
4. **Viral Replication Dynamics and Zebrafish Immune Response:**
   - Viral replication was prominent in the intestine and hematopoietic tissues, supporting the notion of dual tropism in HuNoV infections.
   - The zebrafish immune response, including the upregulation of interferon and related genes, mirrored that seen in human infections, validating the model for studying immune responses.
5. **Influence of Environmental Factors, Like Oxygen Levels, on Viral Infection in Zebrafish:**
   - While the document does not explicitly discuss oxygen levels, environmental factors such as water quality and temperature could influence viral replication and immune responses in zebrafish.
6. **Role of Gut Microbiota in Viral Replication within Zebrafish:**
   - The zebrafish gut microbiota serves functions analogous to those in mammals and may influence viral replication dynamics. The presence of a commensal microbiota is noted, but specific interactions with HuNoV were not detailed in the document.
7. **Application of Zebrafish for Antiviral Drug Screening and Toxicity Testing:**
   - Zebrafish larvae provide a high-throughput platform for screening potential antiviral compounds. The study demonstrated that antiviral treatment significantly reduced viral replication, highlighting the model's utility in drug development.
   - The model allows for the testing of small molecules by adding them to the swimming water, requiring only minute amounts of compounds compared to mammalian studies.
8. **Limitations of Zebrafish as a Model and Future Research Directions:**
   - While zebrafish offer many advantages, there are inherent differences between fish and human physiology that may limit direct translatability of findings.
   - Future research could focus on enhancing the model's relevance to human disease, exploring genetic manipulations, and expanding the study to other foodborne viruses.

Overall, zebrafish provide a promising model for studying foodborne viruses like HuNoV, offering insights into viral pathogenesis, immune responses, and potential therapeutic interventions.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### **V. Evaluation & Analysis**

**1. Accuracy**

- **Content to Evaluate:** Evaluate the accuracy of the generated summary in capturing the core content of the paper, such as main contributions, methods, and research results.
- **Measurement Method:**
  - **Manual Evaluation:** Engage field experts to rate the generated summaries to determine if they effectively and comprehensively capture the key information of the paper.
  - **ROUGE/L BLEU Scores:** Use standard text summarization quality metrics like ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) to assess the similarity between generated summaries and human references.

**2. Semantic Coherence**

- **Content to Evaluate:** Assess whether the generated summary is coherent, maintains the logical structure of the original paper, and has reasonable connections between sentences.
- **Measurement Method:**
  - **Semantic Similarity Analysis:** Use BERT or similar language models to quantify the semantic distance between the generated summary and the original text.
  - **Manual Scoring:** Experts score the generated text based on its coherence and logical completeness.

**3. Compression Ratio**

- **Content to Evaluate:** Evaluate the efficiency of summarizing the original text and whether the key content was effectively retained during compression.
- **Measurement Method:**
  - **Compression Ratio** = Number of words in the summary / Number of words in the original document.
  - Compare if different compression ratios still effectively convey the key information of the paper.

**4. Time Efficiency**

- **Content to Evaluate:** Test the time required by the code to process papers of varying scales and analyze processing speed efficiency, particularly for long documents.
- **Measurement Method:**
  - **Time Statistics Chart:** Record and draw a chart of the time required for the entire process, from file upload and text extraction to generating the summary, especially for comparing long and short documents.
  - **Time Complexity:** Test time performance on different numbers of PDFs and analyze time complexity.

**5. User Satisfaction and Usability**

- **Content to Evaluate:** Use surveys or statistical tools to evaluate user satisfaction with the quality of the generated summaries and ease of use of the process.
- **Measurement Method:**
  - **Survey:** Release a survey to users who have used the tool to assess their satisfaction with the summary quality, generation speed, and user interface. Present results through bar or pie charts.
  - **User Interaction Statistics:** Count the number of interactions users have with the system, such as adjusting prompt settings, to evaluate the intuitiveness and usability of the interface.

**6. Comparison with Existing Tools**

- **Content to Evaluate:** Compare this project with other existing tools or methods for paper summarization to assess performance in accuracy, semantic coherence, processing time, etc.
- **Measurement Method:**
  - **Comparison Table:** Compare this project with other existing tools regarding accuracy, compression ratio, generation speed, etc.
  - **Benchmark Testing:** Conduct standardized testing on an existing set of papers across multiple tools, and compare results using tables or line charts.

**7. Error Analysis**

- **Content to Evaluate:** Analyze cases where the generated summary fails, such as inaccurate content, missing information, semantic breaks, etc.
- **Measurement Method:**
  - **Error Statistics:** Experts evaluate and classify the types and number of errors in the generated summaries.
  - **Case Analysis:** Select some typical failure cases, analyze them in depth, and summarize the reasons for failure, such as misunderstanding of domain terms or poor handling of long texts.

**8. Scalability**

- **Content to Evaluate:** Assess the system's ability to handle a large number of papers and its adaptability to papers in different fields.
- **Measurement Method:**
  - **Batch Processing Test:** Input multiple PDFs from different fields into the system in batches to observe system performance and summary quality.
  - **System Response Time Statistics:** Chart the change in system response time as the input quantity increases.

**9. Prompt Effectiveness Analysis**

- **Content to Evaluate:** Assess how different prompt settings impact the generated result, e.g., does a technical detail-focused prompt make the summary more technical, and does a research significance prompt place more emphasis on the background and contributions?
- **Measurement Method:**
  - **Multi-Prompt Test:** Use multiple prompt styles to test the same paper, record the changes in generated summaries, and demonstrate the specific impact of prompts on the generated result.
  - **Effect Scoring:** Experts rate the summaries based on the relevance and depth of the content, generating statistical results.
  - 
**The relevant evaluation results are shown in the table below:**

![评价指标结果](https://github.com/user-attachments/assets/8c5ae2d0-0319-4e91-aaa4-e83eb7b8c83e)

### **VI. Related Work**

This project draws on various tools and resources to achieve its objectives. Key components include:

**•	OpenAI GPT-4o API:** Utilized for processing and summarizing academic papers with its advanced natural language understanding capabilities.
**•	Python Libraries:** Several Python libraries were utilized to ensure smooth implementation of the project. openai was used for accessing the GPT-4o API, while pdfplumber enabled efficient text extraction from PDF documents. Additionally, standard libraries such as os and google.colab were employed for file handling and enabling file uploads within the Colab environment.
•	Academic Search Engines: Google Scholar served as the primary source for collecting academic papers for dataset creation.
**•	Related Studies: **Insights were drawn from existing research that evaluates the potential of LLMs in academic writing and summarization tasks, providing a foundation for comparison and evaluation.

### **VII. Conclusion**

•**	(1)** The development of this paper review assistant demonstrates the potential of leveraging LLM technology, specifically GPT-4o, to streamline the process of summarizing academic papers. By integrating the GPT-4o API, we successfully addressed the challenges of reading and analyzing lengthy documents, enabling accurate and prompt-focused summaries.

•**	(2)** The results highlight the assistant's efficiency and accuracy, proving its utility for researchers managing large volumes of academic literature. Future efforts can focus on refining prompt customization and expanding the system's adaptability to different academic fields, further enhancing its applicability across diverse research domains.


