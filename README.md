# NeuReview AI | Fake Product Review Detection

![R](https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white) ![Shiny](https://img.shields.io/badge/Shiny-00D2FF?style=for-the-badge) ![Machine Learning](https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge) ![NLP](https://img.shields.io/badge/NLP-38EF7D?style=for-the-badge)

**🔴 Live Demo:** [View the Dashboard on shinyapps.io](https://zafar.shinyapps.io/fakereviewdetectionproject/)

A fully interactive, dark-themed **R Shiny Dashboard** designed to detect fraudulent e-commerce product reviews using a combination of **Natural Language Processing (NLP)** and **Behavioral Machine Learning**.

## ✨ Features

- **🧠 NLP Engine:** Advanced text preprocessing that extracts morphological features, TF-IDF representations, and structural markers (like exclamation frequency and capitalization ratios) to identify artificially generated text.
- **🕸️ Behavioral Machine Learning:** Analyzes user metadata such as review frequency, rating variance, and spatiotemporal clustering to detect coordinated review manipulation campaigns.
- **📊 Trust Score Dashboard:** A dynamic, Power BI-style interactive dashboard built with `plotly` that translates complex ML outputs into actionable business intelligence (Overall Trust Score, Fake vs Genuine Distribution, Rating Variance).
- **📂 Easy Data Ingestion:** Upload standard CSV files. The pipeline automatically sanitizes the data and maps columns dynamically.

## 🚀 Getting Started

### Prerequisites
You will need **R** and optionally **RStudio** installed on your machine. 

Ensure you have the following R packages installed before running the application:
```R
install.packages(c("shiny", "bslib", "dplyr", "ggplot2", "plotly", "DT", "tm", "SnowballC", "randomForest", "caret", "lubridate", "tidytext"))
```

### Running the App Locally

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Zafar7363/FakeReviewDetectionProject.git
   ```
2. Navigate to the project directory and open `app.R` in RStudio.
3. Click the **"Run App"** button in RStudio, or run the following command in your R console:
   ```R
   shiny::runApp("app.R")
   ```
4. Once the app launches in your browser, upload the provided `test_dataset.csv` (or your own dataset) to see the pipeline in action.

## 📁 Dataset Requirements

The application accepts standard `.csv` files. For optimal performance, the dataset should ideally contain the following columns:
- `review_id`
- `product_id`
- `user_id`
- `review_text` (or `text_`)
- `rating` (or `overall`)
- `review_date`

*(Note: The system contains fallback logic to automatically mock or synthesize missing data columns if they are not provided).*

## 🏗️ Architecture

This application simulates a rapid inference pipeline:
1. **Data Ingestion:** Upload CSV via UI.
2. **Feature Engineering:** `tm` and `tidytext` tokenize and extract semantics.
3. **Behavioral Analytics:** Grouping by temporal metadata to detect anomalies.
4. **Classification Engine:** Evaluates a unified feature space against an ML proxy function.
5. **Insights Generation:** Results are aggregated into an empirical **Trust Score**.

---
*Developed for Data Analytics & Machine Learning Academic Project*
