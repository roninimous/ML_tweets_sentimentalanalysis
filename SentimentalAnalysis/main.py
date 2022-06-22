from flask import Flask, render_template, request, redirect, session
import mysql.connector
from sentiments import second
import os


app = Flask(__name__)

# initializing the user cookie
app.secret_key = os.urandom(24)

# blueprint to call the second python file in the project.
app.register_blueprint(second)

# establishing a connection with mysql database made in xampp
try:
    conn = mysql.connector.connect(
        host="localhost", user="root", password="", database="users"
    )
    cursor = conn.cursor()
except:
    @app.route("/")
    def dberror():
        print("An exception occured, Could not connect to the database")
        return render_template("dberror.html")



# call the login template when the url is http://localhost:5000/
@app.route("/")
def login():
    return render_template("login.html")


# call the register template when the url is http://localhost:5000/register
@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/home")
def home():
    if "user_id" in session:
        return render_template("home.html")
    else:
        return redirect("/")


@app.route("/login_validation", methods=["POST"])
def login_validation():
    email = request.form.get("email")
    password = request.form.get("password")
    error_message = "Your email or password is incorrect!"
    cursor.execute(
        """SELECT * from `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(
            email, password
        )
    )
    users = cursor.fetchall()
    # check if a user has already logged in
    if len(users) > 0:
        session["user_id"] = users[0][0].upper()
        return redirect("/home")
    else:
        return render_template("login.html", error_message=error_message)


@app.route("/add_user", methods=["POST"])
def add_user():

    # get user login data and pass the data to database
    name = request.form.get("uname")
    email = request.form.get("uemail")
    password = request.form.get("upassword")

    error_message = "An account with email is already existed!"
    cursor.execute(
        """SELECT * from `users` WHERE `email` LIKE '{}'""".format(
            email)
    )
    existemail = cursor.fetchall()
    # check if a user has already logged in
    if len(existemail) > 0:
        return render_template("register.html", error_message=error_message)
    else:
        cursor.execute(
            """INSERT INTO `users` (`name`,`email`,`password`) VALUES ('{}','{}','{}')""".format(
                name, email, password
            )
        )
        conn.commit()
        cursor.execute("""SELECT * from `users` WHERE `email` LIKE '{}'""".format(email))
        myuser = cursor.fetchall()
        session["user_id"] = myuser[0][0].upper()
        return redirect("/home")


@app.route("/logout")
def logout():
    # close the session
    try:
        session.pop("user_id")
        return redirect("/")
    except:
        return redirect("/")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
