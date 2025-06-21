from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import pandas as pd
import time
import os

def init_browser(download_dir):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    prefs = {"download.default_directory": download_dir}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=options)
    return driver

def download_data_for_date(driver, date, download_dir):
    formatted_date = date.strftime('%Y-%m-%d')
    custom_report_url = f"https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=y&type=c%2C4%2C6%2C11%2C12%2C13%2C21%2C-1%2C34%2C35%2C40%2C41%2C-1%2C23%2C37%2C38%2C50%2C317%2C61%2C-1%2C111%2C-1%2C203%2C199%2C58%2C-1%2C0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%2C12%2C13%2C14%2C15%2C16%2C17%2C18%2C19%2C20%2C21%2C22%2C23%2C24%2C25%2C26%2C27%2C28%2C29%2C30%2C31%2C32%2C33%2C34%2C35%2C36%2C37%2C38%2C39%2C40%2C41%2C42%2C43%2C44%2C45%2C46%2C47%2C48%2C49%2C50%2C51%2C52%2C53%2C54%2C55%2C56%2C57%2C58%2C59%2C60%2C61%2C62%2C63%2C64%2C65%2C66%2C67%2C68%2C69%2C70%2C71%2C72%2C73%2C74%2C75%2C76%2C77%2C78%2C79%2C80%2C81%2C82%2C83%2C84%2C85%2C86%2C87%2C88%2C89%2C90%2C91%2C92%2C93%2C94%2C95%2C96%2C97%2C98%2C99%2C100%2C101%2C102%2C103%2C104%2C105%2C106%2C107%2C108%2C109%2C110%2C111%2C112%2C113%2C114%2C115%2C116%2C117%2C118%2C119%2C120%2C121%2C122%2C123%2C124%2C125%2C126%2C127%2C128%2C129%2C130%2C131%2C132%2C133%2C134%2C135%2C136%2C137%2C138%2C139%2C140%2C141%2C142%2C143%2C144%2C145%2C146%2C147%2C148%2C149%2C150%2C151%2C152%2C153%2C154%2C155%2C156%2C157%2C158%2C159%2C160%2C161%2C162%2C163%2C164%2C165%2C166%2C167%2C168%2C169%2C170%2C171%2C172%2C173%2C174&season={date.year}&month=1000&season1={date.year}&ind=0&startdate={formatted_date}&enddate={formatted_date}&team=0&v_cr=202301"

    driver.get(custom_report_url)
    time.sleep(5)

    try:
        # Click the export data button
        export_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.data-export'))
        )
        export_button.click()
        time.sleep(15)  # Increase wait time for the file to download
    except Exception as e:
        print(f"Error during data download for {formatted_date}: {e}")
        return None

    # Get the latest downloaded file
    latest_file = max([f for f in os.listdir(download_dir)], key=lambda x: os.path.getctime(os.path.join(download_dir, x)))
    latest_file_path = os.path.join(download_dir, latest_file)
    
    print(f"Downloaded file for date: {formatted_date}")
    return latest_file_path, formatted_date

def append_data_to_daily_file(file_path, game_date, save_dir):
    data = pd.read_csv(file_path)
    data['game_date'] = game_date  # Add the game date to each row
    
    filename = os.path.join(save_dir, f"game_logs_{game_date}.csv")
    
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    else:
        updated_data = data

    updated_data.to_csv(filename, index=False)
    print(f"Data for {game_date} saved as {filename}")

def main():
    start_date = datetime.strptime(input("Enter the start date (YYYY-MM-DD): "), '%Y-%m-%d')
    end_date = datetime.strptime(input("Enter the end date (YYYY-MM-DD): "), '%Y-%m-%d')

    download_dir = os.path.expanduser("~/Downloads")
    save_dir = os.path.expanduser("~/FangraphsDailyLogs")  # Directory to save daily game logs
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    driver = init_browser(download_dir)
    driver.get("https://www.fangraphs.com/")
    input("Please log in to Fangraphs in the opened browser and press Enter to continue...")

    print("Proceeding with data download...")

    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    downloaded_dates = []
    for date in date_range:
        result = download_data_for_date(driver, date, download_dir)
        if result:
            file_path, game_date = result
            append_data_to_daily_file(file_path, game_date, save_dir)
            downloaded_dates.append(game_date)

    driver.quit()

    # Save downloaded dates to a text file
    with open("downloaded_dates.txt", "w") as file:
        for date in downloaded_dates:
            file.write(f"{date}\n")
    print("Dates of downloaded files have been recorded in downloaded_dates.txt")

if __name__ == "__main__":
    main()