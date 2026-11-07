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
                    
                    // 1. Check if scrolled to the absolute bottom or top of the container
                    const isAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 2;
                    const isAtTop = container.scrollTop <= 2; // 👈 ADD THIS: detects when scroll is at the top
                    
                    let targetRowIndex = 0;

                    if (isAtBottom) {
                        // If at the bottom, explicitly target the very last row of the current chunk
                        targetRowIndex = rows.length - 1;
                    } else if (isAtTop) {
                        // 👈 ADD THIS: If at the top, explicitly target the very first row (index 0)
                        targetRowIndex = 0; 
                    } else {
                        // Otherwise, find the bottom-most visible row inside the viewport container
                        rows.forEach((row, idx) => {
                            const rect = row.getBoundingClientRect();
                            // Check if the row is within the visible vertical bounds of the container
                            if (rect.top < containerRect.bottom && rect.bottom > containerRect.top) {
                                targetRowIndex = idx; // Continues updating until it hits the last visible row in view
                            }
                        });
                    }

                    const baseOffset = parseInt(container.getAttribute('data-viewport-offset') || '0', 10);
                    const currentRow = baseOffset + targetRowIndex + 1;

                    // Update all matching inputs (Row # field)
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