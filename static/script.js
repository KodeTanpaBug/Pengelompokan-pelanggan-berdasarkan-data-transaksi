function validateFileInput() {
    const inputFile = document.getElementById('formFile');
    const filePath = inputFile.value;
    const allowedExtensions = /(.csv)$/i;
    const errorDiv = document.getElementById('fileError');
    if (!allowedExtensions.exec(filePath)) {
        errorDiv.textContent = 'Mohon upload file dengan format CSV saja.';
        inputFile.value = '';
        return false;
    }
    errorDiv.textContent = '';
    return true;
}

function updateFileName() {
    const inputFile = document.getElementById('formFile');
    const fileHelp = document.getElementById('fileHelp');
    if (inputFile.files.length > 0) {
        fileHelp.textContent = `File yang dipilih: ${inputFile.files[0].name}`;
    } else {
        fileHelp.textContent = '';
    }
}