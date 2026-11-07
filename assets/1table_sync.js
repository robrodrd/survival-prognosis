let currentContainer = null;

function initTableSync() {
    const table = document.querySelector('table');
    if (!table) return false;

    let container = table.parentElement;
    while (container && container !== document.body) {
        const style = window.getComputedStyle(container);
        if (style.overflowY === 'auto' || style.overflowY === 'scroll' || container.id === 'inspector-table-container') {
            
            // If a new dataset re-rendered the container, attach the scroll listener fresh
            if (currentContainer !== container) {
                currentContainer = container;
                
                container.addEventListener('scroll', () => {
                    const tbody = container.querySelector('tbody');
                    if (!tbody) return;
                    const rows = tbody.querySelectorAll('tr');
                    if (rows.length === 0) return;

                    const containerRect = container.getBoundingClientRect();
                    let closestRowIndex = 0;
                    let minDistance = Infinity;

                    rows.forEach((row, idx) => {
                        const rect = row.getBoundingClientRect();
                        const distance = Math.abs(rect.top - containerRect.top);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestRowIndex = idx;
                        }
                    });

                    const baseOffset = parseInt(container.getAttribute('data-viewport-offset') || '0', 10);
                    const currentRow = baseOffset + closestRowIndex + 1;

                    const inputs = document.querySelectorAll('input');
                    inputs.forEach(input => {
                        const placeholder = input.placeholder || '';
                        const wrapper = input.closest('div') || input.parentElement;
                        const wrapperText = wrapper ? wrapper.innerText : '';
                        
                        if (placeholder.includes('Row') || wrapperText.includes('Go') || wrapperText.includes('Type to Filter')) {
                            if (input.type === 'text' || input.type === 'number') {
                                const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                                if (nativeSetter) {
                                    nativeSetter.call(input, currentRow);
                                } else {
                                    input.value = currentRow;
                                }
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                input.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        }
                    });
                });
            }
            return true;
        }
        container = container.parentElement;
    }
    return false;
}

// Keep checking continuously so it re-attaches automatically when datasets are switched
setInterval(initTableSync, 500);