"""
Driver Service
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class DriverService:
	def __init__(self, headless: bool = False):
		self.headless = headless

	def get_driver(self) -> webdriver.Chrome:
		"""
		Sets up and returns a Selenium WebDriver instance.

		Args:
		    headless (bool): Whether to run browser in headless mode

		Returns:
		    webdriver.Chrome: Configured Chrome WebDriver instance
		"""
		# Configure Chrome options
		chrome_options = Options()
		if self.headless:
			chrome_options.add_argument('--headless')

		# Disable automation flags
		chrome_options.add_argument('--disable-blink-features=AutomationControlled')
		chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
		chrome_options.add_experimental_option('useAutomationExtension', False)

		# Initialize the Chrome driver
		driver = webdriver.Chrome(
			service=Service(ChromeDriverManager().install()), options=chrome_options
		)

		return driver
