import React from 'react';
import TextField from 'material-ui/TextField'
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
export default class Home extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      question1: '',
      question2: ''
    }
  }
  render() {

    return (
    <MuiThemeProvider>
     <div style={{textAlign: 'center'}}>
       <TextField hintText="First Question" />
       <TextField hintText="Second Question" value={this.state.question2}
         onChange={this._changeQuestion()}/>

      </div>
      </MuiThemeProvider>
    )
  }
  _changeQuestion(event) {
    console.log(event)
    //this.setState({question2:value})
  }
}
