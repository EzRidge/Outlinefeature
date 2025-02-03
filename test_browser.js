const puppeteer = require('puppeteer');

(async () => {
    try {
        console.log('Launching browser with recommended settings...');
        const browser = await puppeteer.launch({
            headless: true,
            args: [
                '--lang=en-US',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ]
        });

        console.log('Browser launched successfully');
        console.log('Opening paper page...');
        
        const page = await browser.newPage();
        await page.goto('https://www.mdpi.com/2072-4292/11/19/2219', {
            waitUntil: 'networkidle0'
        });
        
        console.log('Page loaded successfully');
        
        // Look for relevant content in the paper
        const content = await page.evaluate(() => {
            const safeText = (element) => {
                if (!element) return '';
                const text = element.innerText || element.textContent;
                return text ? text.trim() : '';
            };

            const safeQuerySelector = (selector) => {
                const element = document.querySelector(selector);
                return element ? safeText(element) : '';
            };

            const safeQuerySelectorAll = (selector) => {
                return Array.from(document.querySelectorAll(selector) || []);
            };

            // Get abstract
            const abstract = safeQuerySelector('.art-abstract');
            
            // Get sections that might contain model information
            const modelKeywords = ['cnn', 'network', 'model', 'architecture', 'implementation', 'code'];
            const relevantSections = safeQuerySelectorAll('section')
                .map(section => safeText(section))
                .filter(text => text && modelKeywords.some(keyword => 
                    text.toLowerCase().includes(keyword)))
                .filter(text => text.length < 2000);
            
            // Get supplementary materials links
            const supplementaryLinks = safeQuerySelectorAll('a[href*="supplementary"]')
                .map(a => ({
                    text: safeText(a),
                    href: a.href
                }))
                .filter(link => link.text);
            
            return {
                abstract,
                relevantSections,
                supplementaryLinks
            };
        });
        
        console.log('\nPaper Analysis:');
        
        if (content.abstract) {
            console.log('\nAbstract:');
            console.log(content.abstract);
        }
        
        if (content.relevantSections.length > 0) {
            console.log('\nRelevant Sections:');
            content.relevantSections.forEach((section, i) => {
                console.log(`\n--- Section ${i + 1} ---`);
                console.log(section);
            });
        }
        
        if (content.supplementaryLinks.length > 0) {
            console.log('\nSupplementary Materials:');
            content.supplementaryLinks.forEach(link => {
                console.log(`- ${link.text}: ${link.href}`);
            });
        }
        
        // Take a screenshot for verification
        await page.screenshot({ path: 'paper_screenshot.png', fullPage: true });
        console.log('\nScreenshot saved as paper_screenshot.png');
        
        await browser.close();
        console.log('\nBrowser closed successfully');
        
    } catch (error) {
        console.error('Error occurred:', error);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
    }
})();