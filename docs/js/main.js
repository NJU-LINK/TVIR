/**
 * TVIR Project Website - Main JavaScript
 */

document.addEventListener("DOMContentLoaded", function () {
  // Initialize AOS (Animate on Scroll)
  AOS.init({
    duration: 800,
    easing: "ease-out-cubic",
    once: true,
    offset: 50,
  });

  // Navigation functionality
  initNavigation();

  // Back to top button
  initBackToTop();

  // Smooth scroll for anchor links
  initSmoothScroll();

  // Copy BibTeX functionality
  initCopyBibtex();

  // Navbar scroll effect
  initNavbarScroll();
});

/**
 * Navigation Menu Toggle (Mobile)
 */
function initNavigation() {
  const navToggle = document.querySelector(".nav-toggle");
  const navMenu = document.querySelector(".nav-menu");

  if (navToggle && navMenu) {
    navToggle.addEventListener("click", function () {
      navMenu.classList.toggle("active");

      // Toggle icon
      const icon = navToggle.querySelector("i");
      if (navMenu.classList.contains("active")) {
        icon.classList.remove("fa-bars");
        icon.classList.add("fa-times");
      } else {
        icon.classList.remove("fa-times");
        icon.classList.add("fa-bars");
      }
    });

    // Close menu when clicking on a link
    const navLinks = navMenu.querySelectorAll("a");
    navLinks.forEach((link) => {
      link.addEventListener("click", function () {
        navMenu.classList.remove("active");
        const icon = navToggle.querySelector("i");
        icon.classList.remove("fa-times");
        icon.classList.add("fa-bars");
      });
    });

    // Close menu when clicking outside
    document.addEventListener("click", function (e) {
      if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
        navMenu.classList.remove("active");
        const icon = navToggle.querySelector("i");
        icon.classList.remove("fa-times");
        icon.classList.add("fa-bars");
      }
    });
  }
}

/**
 * Navbar Scroll Effect
 */
function initNavbarScroll() {
  const navbar = document.querySelector(".navbar");

  if (navbar) {
    window.addEventListener("scroll", function () {
      if (window.scrollY > 50) {
        navbar.classList.add("scrolled");
      } else {
        navbar.classList.remove("scrolled");
      }
    });
  }
}

/**
 * Back to Top Button
 */
function initBackToTop() {
  const backToTopBtn = document.getElementById("backToTop");

  if (backToTopBtn) {
    // Show/hide button based on scroll position
    window.addEventListener("scroll", function () {
      if (window.scrollY > 500) {
        backToTopBtn.classList.add("visible");
      } else {
        backToTopBtn.classList.remove("visible");
      }
    });

    // Scroll to top on click
    backToTopBtn.addEventListener("click", function () {
      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    });
  }
}

/**
 * Smooth Scroll for Anchor Links
 */
function initSmoothScroll() {
  const anchorLinks = document.querySelectorAll('a[href^="#"]');

  anchorLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      const href = this.getAttribute("href");

      if (href !== "#") {
        e.preventDefault();

        const target = document.querySelector(href);
        if (target) {
          const navbarHeight = document.querySelector(".navbar").offsetHeight;
          const targetPosition =
            target.getBoundingClientRect().top +
            window.scrollY -
            navbarHeight -
            20;

          window.scrollTo({
            top: targetPosition,
            behavior: "smooth",
          });
        }
      }
    });
  });
}

/**
 * Copy BibTeX to Clipboard
 */
function initCopyBibtex() {
  // This function is called from the HTML onclick attribute
}

function copyBibtex() {
  const bibtexCode = document.querySelector(".citation-box code");
  const copyBtn = document.querySelector(".copy-btn");

  if (bibtexCode && copyBtn) {
    const text = bibtexCode.textContent;

    navigator.clipboard
      .writeText(text)
      .then(function () {
        // Success feedback
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.classList.add("copied");

        setTimeout(function () {
          copyBtn.innerHTML = originalHTML;
          copyBtn.classList.remove("copied");
        }, 2000);
      })
      .catch(function (err) {
        console.error("Failed to copy: ", err);

        // Fallback for older browsers
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.left = "-9999px";
        document.body.appendChild(textArea);
        textArea.select();

        try {
          document.execCommand("copy");
          const originalHTML = copyBtn.innerHTML;
          copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
          copyBtn.classList.add("copied");

          setTimeout(function () {
            copyBtn.innerHTML = originalHTML;
            copyBtn.classList.remove("copied");
          }, 2000);
        } catch (e) {
          console.error("Fallback copy failed: ", e);
        }

        document.body.removeChild(textArea);
      });
  }
}

/**
 * Lazy Load Images
 */
function initLazyLoad() {
  const images = document.querySelectorAll("img[data-src]");

  if ("IntersectionObserver" in window) {
    const imageObserver = new IntersectionObserver(
      function (entries, observer) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.removeAttribute("data-src");
            imageObserver.unobserve(img);
          }
        });
      },
      {
        rootMargin: "50px 0px",
      },
    );

    images.forEach(function (img) {
      imageObserver.observe(img);
    });
  } else {
    // Fallback for older browsers
    images.forEach(function (img) {
      img.src = img.dataset.src;
      img.removeAttribute("data-src");
    });
  }
}

/**
 * Table Sorting (for results table)
 */
function sortTable(tableId, columnIndex) {
  const table = document.getElementById(tableId);
  if (!table) return;

  const tbody = table.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));

  const isNumeric = !isNaN(parseFloat(rows[0].cells[columnIndex].textContent));

  rows.sort(function (a, b) {
    const aValue = a.cells[columnIndex].textContent.trim();
    const bValue = b.cells[columnIndex].textContent.trim();

    if (isNumeric) {
      return parseFloat(bValue) - parseFloat(aValue); // Descending for numbers
    } else {
      return aValue.localeCompare(bValue);
    }
  });

  rows.forEach(function (row) {
    tbody.appendChild(row);
  });
}

/**
 * Highlight Active Navigation Link
 */
function initActiveNavHighlight() {
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-menu a");

  window.addEventListener("scroll", function () {
    let current = "";
    const navbarHeight = document.querySelector(".navbar").offsetHeight;

    sections.forEach(function (section) {
      const sectionTop = section.offsetTop - navbarHeight - 100;
      const sectionHeight = section.offsetHeight;

      if (
        window.scrollY >= sectionTop &&
        window.scrollY < sectionTop + sectionHeight
      ) {
        current = section.getAttribute("id");
      }
    });

    navLinks.forEach(function (link) {
      link.classList.remove("active");
      if (link.getAttribute("href") === "#" + current) {
        link.classList.add("active");
      }
    });
  });
}

// Initialize active nav highlight
document.addEventListener("DOMContentLoaded", initActiveNavHighlight);

/**
 * Animate Numbers on Scroll
 */
function animateNumbers() {
  const statNumbers = document.querySelectorAll(".stat-number, .stat-value");

  const observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          const target = entry.target;
          const finalValue = target.textContent;

          // Check if it's a simple number
          if (/^\d+$/.test(finalValue)) {
            const endValue = parseInt(finalValue);
            animateValue(target, 0, endValue, 1500);
          }

          observer.unobserve(target);
        }
      });
    },
    {
      threshold: 0.5,
    },
  );

  statNumbers.forEach(function (num) {
    observer.observe(num);
  });
}

function animateValue(element, start, end, duration) {
  const range = end - start;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // Easing function (ease-out)
    const easeOut = 1 - Math.pow(1 - progress, 3);

    const current = Math.round(start + range * easeOut);
    element.textContent = current;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// Initialize number animation
document.addEventListener("DOMContentLoaded", animateNumbers);

/**
 * Image Error Handling
 */
document.addEventListener("DOMContentLoaded", function () {
  const images = document.querySelectorAll(".figure-img");

  images.forEach(function (img) {
    img.addEventListener("error", function () {
      this.style.display = "none";

      // Create placeholder
      const placeholder = document.createElement("div");
      placeholder.className = "image-placeholder";
      placeholder.innerHTML =
        '<i class="fas fa-image"></i><p>Image not available</p>';
      placeholder.style.cssText = `
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 200px;
                background: #f3f4f6;
                border-radius: 0.75rem;
                color: #9ca3af;
                font-size: 1rem;
            `;

      this.parentNode.insertBefore(placeholder, this);
    });
  });
});
