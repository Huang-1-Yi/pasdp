(() => {
  const button = document.querySelector('[data-copy-target]');
  const status = document.querySelector('.copy-status');

  if (!button || !status) return;

  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    const text = target?.innerText.trim();

    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
      status.textContent = 'Citation copied to clipboard.';
    } catch {
      status.textContent = 'Copy was blocked. Select the citation text manually.';
    }

    window.setTimeout(() => {
      button.textContent = 'Copy citation';
      status.textContent = '';
    }, 2400);
  });
})();
