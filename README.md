# Chile Water Quality Dashboard
## Overview
This repository contains a data science dashboard based in Dash designed to visualize various parameters of water quality in Chile. The dashboard includes insightful plots against time and informative histograms, offering a comprehensive view of key water quality metrics.
## Features
- **Time Series Plot:** Track changes in water quality parameters over time, providing a dynamic perspective on trends and patterns.

- **Histograms:** Explore the distribution of different water quality parameters through histograms, facilitating a deeper understanding of their frequency and concentration.
## Data Sources
The dashboard is powered by data collected from the regulator state agency of water sanitary enterprises in Chile (SISS), you can check it following the link: https://www.siss.gob.cl/586/w3-propertyvalue-6405.html
## Usage
Run `src/app.py` and navigate to http://127.0.0.1:8050/ in your browser.
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install dependencies using your preferred package manager.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard application.
   Run `src/app.py` and navigate to. http://127.0.0.1:8050/ in your browser.

## Running the app

### Parameter, Date Range and City Selection
1. **Choose Parameter:** Start by selecting a specific water quality parameter from the dropdown menu.

2. **Specify Date Range:** Define the time frame you want to explore by entering the start and end dates. The trends are created within the selected period.

3. **Select Cities:** Pick one or more cities of interest from the available options. This allows you to narrow down the analysis to specific geographic regions.

### Data Visualization
4. **Plot and Histogram:** In the Dash functionality. Click, Select, Drag and Zoom are allowed.
Visualize the selected parameter's variation over time for the chosen cities and review his statistical behavior within the data range.

### Statistical Summary
5. **Summary Table:** Also, a statistical summary table showcasing key metrics for the selected parameter and cities. This summary provides a quick overview of the dataset's central tendencies.

### Interact and Explore
6. **Iterate and Refine:** Experiment with different parameters, date ranges, and city combinations to tailor the analysis to your specific interests. The dashboard is designed for interactive exploration.

By following these steps, you can harness the power of the dashboard to gain valuable insights into the water quality dynamics of selected cities in Chile.

## Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback to enhance the functionality and usability of the dashboard.

## License
This project is licensed under the [MIT License](LICENSE), allowing for flexibility in sharing and adapting the codebase.

## Contact
For any inquiries or collaboration opportunities, please contact [lplazaalvarez@gmail.com].

Explore and visualize the water quality scenery in Chile with this interactive dashboard!
