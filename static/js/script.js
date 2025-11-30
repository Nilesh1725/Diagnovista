document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // BMI input validation
    const bmiInput = document.getElementById('BMI');
    if (bmiInput) {
        bmiInput.addEventListener('change', function() {
            const bmi = parseFloat(this.value);
            if (bmi > 40) {
                this.classList.add('is-invalid');
                showToast('BMI value seems unusually high. Please double-check.');
            } else if (bmi < 10) {
                this.classList.add('is-invalid');
                showToast('BMI value seems unusually low. Please double-check.');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }
    
    // Symptom search functionality
    const symptomSearch = document.getElementById('symptom-search');
    if (symptomSearch) {
        symptomSearch.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            document.querySelectorAll('.form-check-label').forEach(label => {
                const symptom = label.textContent.toLowerCase();
                const parentDiv = label.closest('.col-md-6');
                if (symptom.includes(searchTerm)) {
                    parentDiv.style.display = 'block';
                } else {
                    parentDiv.style.display = 'none';
                }
            });
        });
    }
});

function showToast(message) {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = 'toast show position-fixed bottom-0 end-0 m-3';
    toast.style.zIndex = '1100';
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="toast-header bg-danger text-white">
            <strong class="me-auto">Warning</strong>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body bg-light">
            ${message}
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
    
    // Add close button functionality
    toast.querySelector('.btn-close').addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    });
}