from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

# Browser options
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-software-rasterizer")
options.add_argument("--window-size=1920,1080")

# Set up ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL for AHL results in English
url = 'https://www.flashscore.ca/baseball/usa/mlb/results/'
driver.get(url)

# Close the cookie banner if it appears
try:
    cookie_banner = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
    )
    cookie_banner.click()
    print("Cookie banner closed.")
except NoSuchElementException:
    print("Cookie banner not found, continuing.")

# Click the "Show more games" button until it's unavailable
while True:
    try:
        load_more_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Show more games')]"))
        )
        load_more_button.click()
        time.sleep(2)  # Wait after clicking
    except (NoSuchElementException, TimeoutException):
        print("All matches loaded or button is unavailable.")
        break
    except ElementClickInterceptedException:
        print("Element not clickable, retrying.")
        time.sleep(2)

# Parse match data
matches = driver.find_elements(By.CSS_SELECTOR, "div.event__match")

# Collect match data
match_data = []

for match in matches:
    # Match time
    match_time = match.find_element(By.CSS_SELECTOR, "div.event__time").text.strip()

    # Team names
    home_team = match.find_element(By.CSS_SELECTOR, "div.event__participant--home").text.strip()
    away_team = match.find_element(By.CSS_SELECTOR, "div.event__participant--away").text.strip()

    # Scores
    scores = match.find_elements(By.CSS_SELECTOR, "div.event__score")
    if len(scores) >= 2:
        home_score = scores[0].text.strip()
        away_score = scores[1].text.strip()
    else:
        home_score = "N/A"
        away_score = "N/A"

    # Clean team names by removing "@" if present
    home_team_cleaned = home_team.replace('@', '').strip()
    away_team_cleaned = away_team.replace('@', '').strip()

    match_info = {
        'Home Team': home_team_cleaned,
        'Home Score': home_score,
        'Visitor Score': away_score,
        'Visiting Team': away_team_cleaned
    }
    match_data.append(match_info)

# Save the collected match data to a CSV file
csv_file_path = 'game_data.csv'  # Updated file name for clarity
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['Home Team', 'Home Score', 'Visitor Score', 'Visiting Team']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()  # Write the header
    for data in match_data:
        writer.writerow(data)  # Write each row

print(f"Match data saved to {csv_file_path}")

# Close the browser
driver.quit()
