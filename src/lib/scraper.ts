import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
import { Mutex } from 'async-mutex';
import os from 'os';
import path from 'path';

class Scraper {
  private static context: any | undefined;
  private static IDLE_KILL_TIMEOUT = 120000;
  private static NAVIGATION_TIMEOUT = 60000;
  private static idleTimeout: NodeJS.Timeout | undefined;
  private static browserMutex = new Mutex();
  private static userCount = 0;

  private static async initBrowser() {
    await this.browserMutex.runExclusive(async () => {
      if (!this.context) {
        const { chromium } = await import('playwright');
        
        // Stealth Persona2 architecture from IP-Web
        const userDataDir = path.join(os.homedir(), '.config', 'vane-stealth-profile');
        
        this.context = await chromium.launchPersistentContext(userDataDir, {
          headless: false, // Visible Chromium window for stealth observation
          args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--disable-infobars',
            '--no-first-run',
            '--no-default-browser-check',
            '--window-size=1920,1080'
          ],
          ignoreDefaultArgs: ['--enable-automation'],
          viewport: { width: 1920, height: 1080 },
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
        });
        
        await this.context.addInitScript(() => {
          Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        });
      }

      if (this.idleTimeout) clearTimeout(this.idleTimeout);
    });
  }

  private static scheduleIdleKill() {
    if (this.idleTimeout) clearTimeout(this.idleTimeout);

    this.idleTimeout = setTimeout(async () => {
      await this.browserMutex.runExclusive(async () => {
        if (this.context && this.userCount === 0) {
          await this.context.close().catch(() => undefined);
          this.context = undefined;
        }
      });
    }, this.IDLE_KILL_TIMEOUT);
  }

  static async scrape(url: string): Promise<{ content: string; title: string }> {
    await this.initBrowser();
    if (!this.context) throw new Error('Browser context not initialized');

    this.userCount++;
    const page = await this.context.newPage(); // Parallel tabs!

    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: this.NAVIGATION_TIMEOUT });
      await page.waitForLoadState('load', { timeout: 10000 }).catch(() => undefined);
      await page.waitForTimeout(2000); // Allow JS to load images and dynamic content

      const html = await page.content();
      const dom = new JSDOM(html, { url });

      // Extract images before readability strips them
      const imgTags = Array.from(dom.window.document.querySelectorAll('img'));
      const imagesMarkdown = imgTags
        .map(img => {
            const src = img.getAttribute('src') || img.getAttribute('data-src');
            return src && src.startsWith('http') ? `![${img.alt || 'image'}](${src})` : '';
        })
        .filter(Boolean)
        .join('\n');

      const article = new Readability(dom.window.document).parse();
      const title = await page.title();
      
      const textContent = article?.textContent?.trim() ?? 'No content available';
      const finalContent = `${textContent}\n\n### Extracted Vision Assets\n${imagesMarkdown}`;

      return {
        content: `# ${title ?? 'No title'} - ${url}\n${finalContent}`,
        title,
      };
    } catch (err) {
      console.log(`Error scraping ${url}:`, err);
      return { title: 'Failed to scrape', content: `# ${url}\n\nError scraping content.` };
    } finally {
      this.userCount--;
      await page.close().catch(() => undefined);
      if (this.userCount === 0) this.scheduleIdleKill();
    }
  }
}
export default Scraper;
