from flask import render_template, request, redirect, url_for, session,flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
from CardioGuardBot import app

app.secret_key = 'hemaaai12341234'

# Enter your database connection details below
app.config['MYSQL_HOST'] = ''
app.config['MYSQL_USER'] = 'hemaa_ai'
app.config['MYSQL_PASSWORD'] = 'hemaa1234' 
app.config['MYSQL_DB'] = 'login_user'

mysql = MySQL(app)

@app.route('/pythonlogin/create_post', methods=['GET', 'POST'])
def create_post():
    if request.method == "POST":
        text = request.form.get('text_cr_post')

        if not text:
            flash('Post cannot be empty', category='error')
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            if 'loggedin' in session:
                #cursor.execute('UPDATE post SET text = %s WHERE author = %s', (text, session['id'],))
                cursor.execute('INSERT INTO post VALUES(NULL, %s, %s, %s)', (text, session['id'], session['username'], ))
                mysql.connection.commit()
                #flash('Post created!', category='success')
                return redirect(url_for('postat'))

    return render_template('blog/create_post.html', username = session['username'], title="Create Post")

#from models import accounts, post, comments, likes

@app.route("/pythonlogin/postat")
def postat():
    #postat = post.query.filter_by(author = 2)
    if 'loggedin' in session:
        user = session['username']

        if not user:
            return redirect(url_for('home'))
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM post')
        posts = cursor.fetchall()
        cursor.execute('SELECT * FROM comments')
        comments = cursor.fetchall()

    return render_template("blog/posts.html", comments = comments, user = session['id'], posts= posts, username=user)

@app.route("/pythonlogin/delete_post/<id>")
def delete_post(id):
    #id = int(request.form("post_id"))
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if 'loggedin' in session:
        cursor.execute('DELETE FROM post WHERE id = %s', (id,))
        mysql.connection.commit()
        
        return redirect(url_for('postat'))

@app.route("/pythonlogin/create_comment/<id>", methods=['GET', 'POST'])
def create_comment(id):
    if request.method == "POST":
        text = request.form.get('text_cr_comment')

        if not text:
            flash('Post cannot be empty', category='error')
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            if 'loggedin' in session:
                cursor.execute('INSERT INTO comments VALUES(NULL, %s, %s, %s)', (text, session['id'], id, ))
                mysql.connection.commit()
    return redirect(url_for('postat'))

@app.route("/pythonlogin/delete_comment/<id>")
def delete_comment(id):
    #id = int(request.form("post_id"))
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if 'loggedin' in session:
        cursor.execute('DELETE FROM comments WHERE id = %s', (id,))
        mysql.connection.commit()
        
        return redirect(url_for('postat'))
    
@app.route("/pythonlogin/delete_user/<name>")
def delete_user(name):
    #id = int(request.form("post_id"))
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if 'loggedin' in session:
        cursor.execute('DELETE FROM accounts WHERE username = %s', (name,))
        mysql.connection.commit()
        
        return redirect(url_for('admin'))

@app.route("/pythonlogin/admin")
def admin():
    if 'loggedin' in session:
        if session['id'] == 1:
            
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM post')
            posts = cursor.fetchall()
            cursor.execute('SELECT * FROM comments')
            comments = cursor.fetchall()
            cursor.execute('SELECT * FROM accounts')
            accounts = cursor.fetchall()
            return render_template('auth/admin.html',comments = comments, accounts = accounts,
                                    posts= posts, username=session['username'], user = session['id'], title="Admin")
    return redirect(url_for('login')) 

@app.route("/pythonlogin/layout")
def layout():
    if 'loggedin' in session:
        return render_template('home/layout.html', username=session['username'],title="Layout")
    return redirect(url_for('login')) 

if __name__ =='__main__':
	app.run(debug=True)