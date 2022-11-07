import React from "react";
import { useNavigate } from "react-router-dom";
import Particle from "../../components/Particle";
import { Store } from "../../classes/storage";

import Typed from "typed.js";
import { useEffect, useRef } from "react";
import { useSearchParams } from 'react-router-dom';
import { Select, Stack } from "@chakra-ui/react";
import "./Home.css";
import logo from "../../assets/vidyabot.png";
const Home = () => {
  const el = useRef(null);

  useEffect(() => {
    const typed = new Typed(el.current, {
      strings: ["Welcome to VidyaBOT", "How can I help you"], // Strings to display
      // Speed settings, try diffrent values until you get good results
      startDelay: 300,
      typeSpeed: 150,

      backSpeed: 50,
      backDelay: 100,
      loop: true,
      showCursor: true,
      cursorChar: "_",
    });

    return () => {
      typed.destroy();
    };
  }, []);

  const navigate = useNavigate();
  const handleLanguageSelect = (event) => {

    Store.languageCode = event.target.value;
    console.log(Store.languageCode);
    navigate("/model");
  };

  // lang is value of select option


  // const [lang, setLang] = useState(false);

  // if (lang === true) {
  //   handleEng();
  // }else if (lang === false) {
  //   handleHin();
  // }


  // const handleChange = (e) => {
  //   if (e.target.value == "en-IN") {
  //     handleEng();
  //   }
  //   else if (e.target.value == "hi-IN") {
  //     handleHin();
  //   }
  //   else if (e.target.value == "mr-IN") {
  //     handleMar();
  //   }
  // }

  const [searchParams] = useSearchParams();
  const uid = searchParams.get("uid");
  Store.uid = uid;


  return (
    <>
      <Particle></Particle>
      <div className="home">
        <div className="row1">
          <img src={logo} alt="MITWPU" />
        </div>
        <div className="row2">
          <h1 className="welcome">
            <span ref={el}></span>
          </h1>
          <br></br>
          <br></br>
          <h4 className="select-text">Select a language</h4>
          <br></br>

          <div className="btn">
            {/* <button onClick={handle}>ENGLISH</button>
            <button onClick={handleHin}>हिन्दी</button>
            <button onClick={handleMar}>मराठी</button> */}
            <Stack spacing={3} className="dropdown">
              <Select variant='filled' style={{
                backgroundColor: "#8A2BE2",
                width: "180px",
                border: "solid 3.5px #8A2BE2",
              }}
                placeholder="Select a language"
                onChange={handleLanguageSelect}
              >
                <option value='en-IN'>English</option>
                <option value='hi-IN'>Hindi</option>
                <option value='mr-IN'>Marathi</option>
                <option value='bn-IN'>Bengali</option>

              </Select>
            </Stack>
          </div>
        </div>
        <div className="row3"></div>
      </div>
    </>
  );
};

export default Home;
