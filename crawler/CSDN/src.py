from playwright.sync_api import Playwright, sync_playwright


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://ask.csdn.net/channel/1005?rewardType&stateType=0&sortBy=1&quick=6&essenceType=1&tagName=essence")
    for i in range(1, 1800):
        page.mouse.wheel(0,10000)
    elements = page.query_selector_all("xpath=//div[@class='question-content-wrapper']//div[@class='title-box']/a")
    for element in elements:
        print(element.get_attribute('href'))
    page.close()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
