const path = require('path');
const { src, dest, series } = require('gulp');
const replace = require('gulp-replace');
const useref = require('gulp-useref');

module.exports = conf => {
  // Copy templatePath html files and {{ url_for('static', filename=' to buildPath
  // -------------------------------------------------------------------------------
  const prodCopyTask = function () {
    return src(`${templatePath}/**/*.html`)
      .pipe(dest(buildPath))
      .pipe(src('{{ url_for('static', filename='/**/*'))
      .pipe(dest(`${buildPath}/{{ url_for('static', filename='/`));
  };

  // Rename {{ url_for('static', filename=' path
  // -------------------------------------------------------------------------------
  const prodRenameTasks = function () {
    return src(`${buildPath}/*.html`)
      .pipe(replace('../{{ url_for('static', filename='', '{{ url_for('static', filename=''))
      .pipe(dest(buildPath))
      .pipe(src(`${buildPath}/{{ url_for('static', filename='/**/*`))
      .pipe(replace('../{{ url_for('static', filename='', '{{ url_for('static', filename=''))
      .pipe(dest(`${buildPath}/{{ url_for('static', filename='/`));
  };

  // Combine js vendor {{ url_for('static', filename=' in single core.js file using UseRef
  // -------------------------------------------------------------------------------
  const prodUseRefTasks = function () {
    return src(`${buildPath}/*.html`).pipe(useref()).pipe(dest(buildPath));
  };

  const prodAllTask = series(prodCopyTask, prodRenameTasks, prodUseRefTasks);

  // Exports
  // ---------------------------------------------------------------------------

  return {
    copy: prodCopyTask,
    rename: prodRenameTasks,
    useref: prodUseRefTasks,
    all: prodAllTask
  };
};
