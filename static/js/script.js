// Custom scripts for Smart Presence
console.log("Smart Presence script loaded.");

// Auto-collapse navbar on link click (mobile)
document.addEventListener('DOMContentLoaded', function () {
  var navbarNav = document.getElementById('navbarNav');
  if (!navbarNav) return;

  navbarNav.addEventListener('click', function (e) {
    var target = e.target;
    if (target.closest('a.nav-link, button.dropdown-item, a.dropdown-item')) {
      var bsCollapse = bootstrap.Collapse.getInstance(navbarNav);
      if (!bsCollapse) {
        bsCollapse = new bootstrap.Collapse(navbarNav, { toggle: false });
      }
      if (window.getComputedStyle(document.querySelector('.navbar-toggler')).display !== 'none') {
        bsCollapse.hide();
      }
    }
  });
});